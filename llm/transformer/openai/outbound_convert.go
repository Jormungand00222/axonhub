package openai

import (
	"context"
	"fmt"
	"maps"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

const (
	// emptyAssistantContentPlaceholder replaces the content of assistant turns
	// that carry none: turns that only request tool calls, turns that only carry
	// reasoning, and turns whose parts were all filtered out (compaction, or a
	// stripped sanitization marker). It cannot be an empty or whitespace-only
	// string, because LiteLLM-family gateways rewrite any blank content into
	// litellmSanitizedEmptyContentMarker, which pollutes the model's context and
	// gets copied into its output. It cannot be null either, because stricter
	// OpenAI-compatible upstreams reject a missing or null content with a 400
	// schema error. A single "." passes those gateways unchanged and models do
	// not mimic it.
	emptyAssistantContentPlaceholder = "."

	// litellmSanitizedEmptyContentMarker is the placeholder LiteLLM-family
	// gateways substitute for blank message content. History replayed from such
	// a gateway carries it as real text, so assistant turns whose content is only
	// this marker are treated as content-less.
	litellmSanitizedEmptyContentMarker = "[System: Empty message content sanitised to satisfy protocol]"
)

// RequestFromLLM creates an OpenAI Request from unified llm.Request with reasoning
// field configuration. When the request has no explicit prompt cache key, ctx's
// session ID is used as a fallback when available.
func RequestFromLLM(ctx context.Context, r *llm.Request, reasoningField ReasoningField) *Request {
	req, _ := requestFromLLM(ctx, r, reasoningField)
	return req
}

func requestFromLLM(ctx context.Context, r *llm.Request, reasoningField ReasoningField) (*Request, error) {
	if r == nil {
		return nil, nil
	}
	// The plain codec cannot represent Responses-only tool lifecycle state.
	// Wrapper transformers call this codec directly, so downgrade here to keep
	// specialized calls from being serialized as invalid Chat history entries
	// (empty names/arguments) that strict providers reject.
	if isResponsesAPIFormat(r.APIFormat) && !r.TransformOptions.DisableResponsesChatCompat {
		var err error
		r, err = shared.DowngradeResponsesChatToolLifecycle(r)
		if err != nil {
			return nil, fmt.Errorf("failed to downgrade Responses tool lifecycle: %w", err)
		}
	}
	req := requestFromLLMBase(ctx, r)
	req.Messages = lo.Map(r.Messages, func(m llm.Message, _ int) Message {
		message := MessageFromLLMWithConfig(m, reasoningField)
		// Tool call indexes identify positions within one assistant request
		// message. Normalize history here without rewriting response deltas,
		// whose indexes must remain stable across streaming chunks.
		for index := range message.ToolCalls {
			message.ToolCalls[index].Index = index
		}
		return message
	})
	req.Messages = mergeSystemMessages(req.Messages)
	req.Tools = lo.FilterMap(r.Tools, func(tool llm.Tool, _ int) (Tool, bool) {
		return ToolFromLLM(tool), tool.Type == llm.ToolTypeFunction
	})
	if r.ToolChoice != nil {
		req.ToolChoice = &ToolChoice{ToolChoice: r.ToolChoice.ToolChoice}
		if r.ToolChoice.NamedToolChoice != nil {
			req.ToolChoice.NamedToolChoice = &NamedToolChoice{
				Type: r.ToolChoice.NamedToolChoice.Type,
				Function: ToolFunction{
					Name: r.ToolChoice.NamedToolChoice.Function.Name,
				},
			}
		}
	}
	if len(req.Tools) == 0 {
		req.ParallelToolCalls = nil
	}
	return req, nil
}

// RequestFromLLMWithResponsesTools converts a request while retaining metadata
// needed to restore Responses-only tool calls. Provider-specific Chat codecs use
// this path when they advertise Responses chat-tool lifecycle support.
func RequestFromLLMWithResponsesTools(
	ctx context.Context,
	r *llm.Request,
	reasoningField ReasoningField,
) (*Request, map[string]any, error) {
	if r == nil {
		return nil, nil, nil
	}
	if !isResponsesAPIFormat(r.APIFormat) {
		req, err := requestFromLLM(ctx, r, reasoningField)
		return req, nil, err
	}
	if r.TransformOptions.DisableResponsesChatCompat {
		req, err := requestFromLLM(ctx, r, reasoningField)
		return req, nil, err
	}

	req, adapter, err := requestFromLLMWithResponsesToolAdapterContext(ctx, r, reasoningField)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", transformer.ErrInvalidRequest, err)
	}
	if adapter == nil {
		return req, nil, nil
	}

	return req, responsesChatToolMetadata(r.TransformerMetadata, adapter), nil
}

func responsesChatToolMetadata(existing map[string]any, adapter *responsesChatToolAdapter) map[string]any {
	if adapter == nil {
		return maps.Clone(existing)
	}

	metadata := make(map[string]any, len(existing)+4)
	maps.Copy(metadata, existing)
	metadata[responsesChatStrictFinishMetadataKey] = true
	if mappings := adapter.mappings(); len(mappings) > 0 {
		metadata[ResponsesChatToolMappingsMetadataKey] = mappings
	}
	if catalog := adapter.catalog(); len(catalog) > 0 {
		metadata[ResponsesChatToolCatalogMetadataKey] = catalog
	}
	if len(adapter.warnings) > 0 {
		metadata[responsesChatToolWarningsMetadataKey] = append([]string(nil), adapter.warnings...)
	}
	return metadata
}

// requestFromLLMBase converts fields shared by plain and Responses-adapted Chat requests.
func requestFromLLMBase(ctx context.Context, r *llm.Request) *Request {
	req := &Request{
		Model:               r.Model,
		FrequencyPenalty:    r.FrequencyPenalty,
		Logprobs:            r.Logprobs,
		MaxCompletionTokens: r.MaxCompletionTokens,
		MaxTokens:           r.MaxTokens,
		PresencePenalty:     r.PresencePenalty,
		Seed:                r.Seed,
		Store:               r.Store,
		Temperature:         r.Temperature,
		TopLogprobs:         r.TopLogprobs,
		TopP:                r.TopP,
		PromptCacheKey:      r.PromptCacheKey,
		SafetyIdentifier:    r.SafetyIdentifier,
		User:                r.User,
		LogitBias:           r.LogitBias,
		Metadata:            r.Metadata,
		Modalities:          r.Modalities,
		ReasoningEffort:     r.ReasoningEffort,
		ServiceTier:         r.ServiceTier,
		Stream:              r.Stream,
		ParallelToolCalls:   r.ParallelToolCalls,
		Verbosity:           r.Verbosity,
	}

	if ctx != nil && lo.FromPtr(req.PromptCacheKey) == "" {
		if sessionID, ok := shared.GetSessionID(ctx); ok && sessionID != "" {
			req.PromptCacheKey = lo.ToPtr(sessionID)
		}
	}

	if r.Stop != nil {
		req.Stop = &Stop{Stop: r.Stop.Stop, MultipleStop: r.Stop.MultipleStop}
	}
	if r.StreamOptions != nil {
		req.StreamOptions = &StreamOptions{IncludeUsage: r.StreamOptions.IncludeUsage}
	}
	if r.ResponseFormat != nil {
		req.ResponseFormat = &ResponseFormat{
			Type: r.ResponseFormat.Type, JSONSchema: r.ResponseFormat.JSONSchema,
		}
	}
	return req
}

// requestFromLLMWithResponsesToolAdapter converts a request and preserves reversible tool mappings.
// It is retained for package-local callers that do not need session-derived fields.
func requestFromLLMWithResponsesToolAdapter(r *llm.Request, reasoningField ReasoningField) (*Request, *responsesChatToolAdapter, error) {
	return requestFromLLMWithResponsesToolAdapterContext(context.Background(), r, reasoningField)
}

func requestFromLLMWithResponsesToolAdapterContext(ctx context.Context, r *llm.Request, reasoningField ReasoningField) (*Request, *responsesChatToolAdapter, error) {
	if r == nil {
		return nil, nil, nil
	}
	toolAdapter := newResponsesChatToolAdapter(r.Tools)
	degradedToolChoice := toolAdapter.degradeUnsupportedRawToolSelector(r)

	req := requestFromLLMBase(ctx, r)

	// Build the callable catalog before converting history so specialized calls
	// resolve through the same stable names as the current tool declarations.
	req.Tools = toolAdapter.filterAllowedTools(toolAdapter.convertTools(r.Tools), r.ToolChoice)

	// Convert messages. Responses can retain assistant-only metadata such as
	// encrypted reasoning or compaction items that Chat Completions cannot
	// represent. Once those fields are stripped, omit the empty assistant
	// message instead of sending an invalid history entry to the provider.
	droppedEmptyAssistants := 0
	req.Messages = lo.FilterMap(r.Messages, func(m llm.Message, _ int) (Message, bool) {
		converted := toolAdapter.convertMessage(m, reasoningField)
		if !hasChatAssistantPayload(converted) {
			droppedEmptyAssistants++
			return Message{}, false
		}
		return converted, true
	})
	req.Messages = mergeSystemMessages(req.Messages)
	if droppedEmptyAssistants > 0 {
		toolAdapter.addWarningf(
			"empty_assistant_message: dropped %d history message(s) with no Chat-compatible payload",
			droppedEmptyAssistants,
		)
	}

	// Convert ToolChoice
	if !degradedToolChoice {
		req.ToolChoice = toolAdapter.convertToolChoice(r.ToolChoice)
	}

	if len(req.Tools) == 0 {
		req.ParallelToolCalls = nil
		if req.ToolChoice != nil && req.ToolChoice.ToolChoice != nil {
			switch *req.ToolChoice.ToolChoice {
			case "auto", "none":
				req.ToolChoice = nil
			case "required":
				toolAdapter.setError(fmt.Errorf("unsupported_tool_choice: required tool choice has no callable tools after Responses-to-Chat conversion"))
			}
		}
	}

	if toolAdapter.err != nil {
		return nil, toolAdapter, toolAdapter.err
	}
	return req, toolAdapter, nil
}

// mergeSystemMessages collapses all system-role messages into one at the
// position of the first, dropping the rest. The Chat Completions spec allows
// a single system message, and strict OpenAI-compatible upstreams reject
// extras with "System message must be at the beginning".
func mergeSystemMessages(msgs []Message) []Message {
	var (
		systemCount int
		firstSystem = -1
		systemText  strings.Builder
	)

	for i, m := range msgs {
		if m.Role != "system" {
			continue
		}
		if firstSystem == -1 {
			firstSystem = i
		}
		systemCount++
		for _, text := range messageTextParts(m) {
			if systemText.Len() > 0 {
				systemText.WriteString("\n\n")
			}
			systemText.WriteString(text)
		}
	}

	if systemCount == 0 {
		return msgs
	}

	merged := make([]Message, 0, len(msgs)-systemCount+1)
	mergedSystem := msgs[firstSystem]
	if systemCount > 1 {
		mergedSystem.Content = MessageContent{Content: lo.ToPtr(systemText.String())}
	}
	merged = append(merged, mergedSystem)
	for _, m := range msgs {
		if m.Role != "system" {
			merged = append(merged, m)
		}
	}
	return merged
}

// messageTextParts extracts the text segments of a message. MultipleContent
// takes precedence over the scalar Content (matching MessageContent.MarshalJSON
// and the documented mutual-exclusivity rule); the scalar is read only when
// there are no parts, so a message carrying both representations is not
// double-counted.
func messageTextParts(m Message) []string {
	if len(m.Content.MultipleContent) > 0 {
		parts := make([]string, 0, len(m.Content.MultipleContent))
		for _, p := range m.Content.MultipleContent {
			if p.Type == "text" && p.Text != nil {
				parts = append(parts, *p.Text)
			}
		}
		return parts
	}
	if m.Content.Content != nil {
		return []string{*m.Content.Content}
	}
	return nil
}

// MessageFromLLM creates OpenAI Message from unified llm.Message.
// Defaults to ReasoningFieldAll to preserve both reasoning fields.
//
// MessageFromLLM serves the response direction: the payload heads back to the
// client, so empty assistant content keeps the historical empty string, which
// client-side content accumulators treat as no content.
func MessageFromLLM(m llm.Message) Message {
	return messageFromLLM(m, ReasoningFieldAll, false)
}

// MessageFromLLMWithConfig creates OpenAI Message from unified llm.Message with reasoning field configuration.
//
// MessageFromLLMWithConfig serves the request direction: the payload heads to
// an upstream provider, so assistant turns get the LiteLLM sanitization marker
// stripped and content-less ones take emptyAssistantContentPlaceholder.
func MessageFromLLMWithConfig(m llm.Message, reasoningField ReasoningField) Message {
	return messageFromLLM(m, reasoningField, true)
}

// messageFromLLM converts a unified message into an OpenAI message.
// forUpstreamRequest selects the request-direction content rules: strip the
// LiteLLM sanitization marker from assistant turns and substitute
// emptyAssistantContentPlaceholder for content-less ones.
func messageFromLLM(m llm.Message, reasoningField ReasoningField, forUpstreamRequest bool) Message {
	var reasoningContent, reasoning *string

	// Apply reasoning field configuration
	switch reasoningField {
	case ReasoningFieldContent:
		// Only use reasoning_content field
		// Prefer ReasoningContent, fallback to Reasoning if ReasoningContent is nil
		reasoningContent = m.ReasoningContent
		if reasoningContent == nil && m.Reasoning != nil {
			reasoningContent = m.Reasoning
		}
		reasoning = nil
	case ReasoningFieldReasoning:
		// Only use reasoning field
		// Prefer Reasoning, fallback to ReasoningContent if Reasoning is nil
		reasoning = m.Reasoning
		if reasoning == nil && m.ReasoningContent != nil {
			reasoning = m.ReasoningContent
		}
		reasoningContent = nil
	case ReasoningFieldNone:
		// Strip all reasoning fields
		reasoningContent = nil
		reasoning = nil
	default: // ReasoningFieldAll
		// Preserve both reasoning fields with sync logic
		reasoningContent = m.ReasoningContent
		reasoning = m.Reasoning

		// Sync: if one field has value and the other is nil/empty, copy the value
		if reasoningContent == nil && reasoning != nil && *reasoning != "" {
			reasoningContent = reasoning
		}
		if reasoning == nil && reasoningContent != nil && *reasoningContent != "" {
			reasoning = reasoningContent
		}
	}

	// Build the Message with determined fields
	msg := Message{
		Role:             m.Role,
		Name:             m.Name,
		Refusal:          m.Refusal,
		ToolCallID:       m.ToolCallID,
		ReasoningContent: reasoningContent,
		Reasoning:        reasoning,
	}

	if m.Audio != nil {
		msg.Audio = &OutputAudio{
			ID:         m.Audio.ID,
			Data:       m.Audio.Data,
			ExpiresAt:  m.Audio.ExpiresAt,
			Transcript: m.Audio.Transcript,
		}
	}

	// Convert Content
	msg.Content = MessageContentFromLLM(m.Content)

	// Chat Completions accepts a string for the tool role; a multi-part tool
	// result (e.g. a Responses function_call_output with several text items)
	// would otherwise serialize as an array and be rejected with a 400.
	if m.Role == "tool" {
		msg.Content = flattenChatToolContent(msg.Content)
	}

	// LiteLLM-family gateways rewrite blank message content into a fixed
	// sanitization marker, which the model then copies into its own output.
	// Normalize assistant turns so no blank content reaches such a gateway, and
	// strip the marker itself from history replayed through one. User turns
	// keep their text: a user may quote the marker on purpose.
	if forUpstreamRequest && msg.Role == "assistant" {
		msg.Content = normalizeAssistantRequestContent(msg.Content)
	}

	// Convert ToolCalls
	if m.ToolCalls != nil {
		msg.ToolCalls = lo.Map(m.ToolCalls, func(tc llm.ToolCall, _ int) ToolCall {
			return ToolCallFromLLM(tc)
		})
	}

	// Assistant turns can carry no content: turns that only request tool calls,
	// turns that only carry reasoning (a Responses reasoning item not followed by
	// text or a call), and messages whose parts were all filtered out (e.g.
	// compaction, or a stripped sanitization marker) are left with an empty part
	// list. These cases would reach the wire as a missing or null content field,
	// which the OpenAI spec permits but stricter OpenAI-compatible upstreams
	// reject because their schema only accepts a string or an array.
	//
	// Requests headed upstream take emptyAssistantContentPlaceholder, because a
	// blank string would be rewritten by LiteLLM-family gateways into their
	// sanitization marker. Responses headed back to the client keep the empty
	// string, which every implementation accepts and client-side accumulators
	// treat as no content.
	if msg.Role == "assistant" && msg.Content.Content == nil && len(msg.Content.MultipleContent) == 0 {
		placeholder := ""
		if forUpstreamRequest {
			placeholder = emptyAssistantContentPlaceholder
		}
		msg.Content = MessageContent{Content: lo.ToPtr(placeholder)}
	}

	// Convert Annotations
	if len(m.Annotations) > 0 {
		msg.Annotations = lo.Map(m.Annotations, func(a llm.Annotation, _ int) Annotation {
			return AnnotationFromLLM(a)
		})
	}

	return msg
}

// AnnotationFromLLM creates OpenAI Annotation from unified llm.Annotation.
func AnnotationFromLLM(a llm.Annotation) Annotation {
	annotation := Annotation{
		Type:       a.Type,
		StartIndex: a.StartIndex,
		EndIndex:   a.EndIndex,
	}

	if a.URLCitation != nil {
		annotation.URLCitation = &URLCitation{
			URL:   a.URLCitation.URL,
			Title: a.URLCitation.Title,
		}
	}

	return annotation
}

// MessageContentFromLLM creates OpenAI MessageContent from unified llm.MessageContent.
func MessageContentFromLLM(c llm.MessageContent) MessageContent {
	content := MessageContent{
		Content: c.Content,
	}

	if c.MultipleContent != nil {
		content.MultipleContent = lo.FilterMap(c.MultipleContent, func(p llm.MessageContentPart, _ int) (MessageContentPart, bool) {
			switch p.Type {
			case "compaction", "compaction_summary":
				return MessageContentPart{}, false
			default:
				return MessageContentPartFromLLM(p), true
			}
		})
	}

	return content
}

// normalizeAssistantRequestContent removes the assistant content shapes that a
// LiteLLM-family gateway would rewrite into litellmSanitizedEmptyContentMarker,
// so the model never receives that protocol text. Two sources produce such
// content, and both are treated as no content:
//
//  1. blank text — an empty or whitespace-only scalar content, and text parts
//     whose text is nil or whitespace-only;
//  2. the sanitization marker itself, replayed as history from a gateway that
//     had already rewritten blank content into it.
//
// Text is compared trimmed so surrounding whitespace added by intermediate
// proxies does not defeat the check, and parts are examined after
// MessageContentFromLLM has dropped the types Chat Completions cannot
// represent. The caller turns an emptied content into
// emptyAssistantContentPlaceholder. MessageContent marshals MultipleContent in
// preference to Content, so a scalar sitting next to a part list is dead
// weight: when every part is dropped, the scalar is cleared too rather than
// resurfacing as visible content.
func normalizeAssistantRequestContent(content MessageContent) MessageContent {
	if content.Content != nil {
		trimmed := strings.TrimSpace(*content.Content)
		if trimmed == "" || trimmed == litellmSanitizedEmptyContentMarker {
			content.Content = nil
		}
	}

	if len(content.MultipleContent) > 0 {
		content.MultipleContent = lo.FilterMap(content.MultipleContent, func(p MessageContentPart, _ int) (MessageContentPart, bool) {
			if p.Type != "text" {
				return p, true
			}
			if p.Text == nil {
				return MessageContentPart{}, false
			}

			trimmed := strings.TrimSpace(*p.Text)
			if trimmed == "" || trimmed == litellmSanitizedEmptyContentMarker {
				return MessageContentPart{}, false
			}

			return p, true
		})

		// The list is the authoritative representation for this turn, so a
		// scalar that marshaling already ignored must not resurface once every
		// part has been dropped.
		if len(content.MultipleContent) == 0 {
			content.Content = nil
		}
	}

	return content
}

// MessageContentPartFromLLM creates OpenAI MessageContentPart from unified llm.MessageContentPart.
func MessageContentPartFromLLM(p llm.MessageContentPart) MessageContentPart {
	part := MessageContentPart{
		Type: normalizeContentPartType(p.Type),
		Text: p.Text,
	}

	if p.ImageURL != nil {
		part.ImageURL = &ImageURL{
			URL:    p.ImageURL.URL,
			Detail: p.ImageURL.Detail,
		}
	}

	if p.VideoURL != nil {
		part.VideoURL = &VideoURL{
			URL: p.VideoURL.URL,
		}
	}

	if p.InputAudio != nil {
		part.InputAudio = &InputAudio{
			Format: p.InputAudio.Format,
			Data:   p.InputAudio.Data,
		}
	}

	if p.Document != nil {
		part.Type = "file"
		part.File = &File{
			FileID:   p.Document.FileID,
			Filename: p.Document.Filename,
		}
		if strings.HasPrefix(p.Document.URL, "data:") {
			part.File.FileData = p.Document.URL
		}
		if part.File.Filename == "" && p.Document.MIMEType == "application/pdf" {
			part.File.Filename = "document.pdf"
		}
	}

	return part
}

func validateChatDocumentParts(messages []llm.Message) error {
	for _, message := range messages {
		for _, part := range message.Content.MultipleContent {
			if part.Type != "document" || part.Document == nil {
				continue
			}
			if part.Document.FileID == "" && part.Document.URL != "" && !strings.HasPrefix(part.Document.URL, "data:") {
				return fmt.Errorf("%w: OpenAI Chat file inputs require file_id or a data URL in file_data", transformer.ErrInvalidRequest)
			}
		}
	}

	return nil
}

// normalizeContentPartType maps Responses-only text part types onto the plain
// "text" type used by Chat Completions. The Responses API distinguishes input
// from output text, but Chat Completions has a single text part type, and strict
// OpenAI-compatible upstreams reject the ones they do not know. Types that Chat
// Completions does understand (image_url, video_url, input_audio, ...) pass through.
func normalizeContentPartType(partType string) string {
	switch partType {
	case "input_text", "output_text":
		return "text"
	default:
		return partType
	}
}

// flattenChatToolContent collapses a Chat tool message's content into a single
// string. Text parts are concatenated in order; parts a Chat tool message
// cannot carry (e.g. images) are dropped, which still beats emitting an array
// that the provider rejects outright.
func flattenChatToolContent(content MessageContent) MessageContent {
	if len(content.MultipleContent) == 0 {
		return content
	}

	var builder strings.Builder
	for _, part := range content.MultipleContent {
		if part.Type == "text" && part.Text != nil {
			builder.WriteString(*part.Text)
		}
	}

	text := builder.String()

	return MessageContent{Content: &text}
}

// ToolFromLLM creates OpenAI Tool from unified llm.Tool.
func ToolFromLLM(t llm.Tool) Tool {
	return Tool{
		Type: t.Type,
		Function: Function{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
			Strict:      t.Function.Strict,
		},
	}
}

// ToolCallFromLLM creates OpenAI ToolCall from unified llm.ToolCall.
func ToolCallFromLLM(tc llm.ToolCall) ToolCall {
	toolCall := ToolCall{
		ID:   tc.ID,
		Type: tc.Type,
		Function: FunctionCall{
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		},
		Index: tc.Index,
	}

	if raw, ok := tc.TransformerMetadata[TransformerMetadataKeyGoogleThoughtSignature].(string); ok && raw != "" {
		toolCall.ExtraContent = &ToolCallExtraContent{
			Google: &ToolCallGoogleExtraContent{
				ThoughtSignature: raw,
			},
		}
	}

	return toolCall
}

// ToLLMResponse converts OpenAI Response to unified llm.Response.
func (r *Response) ToLLMResponse() *llm.Response {
	if r == nil {
		return nil
	}

	resp := &llm.Response{
		ID:                r.ID,
		Object:            r.Object,
		Created:           r.Created,
		Model:             r.Model,
		SystemFingerprint: r.SystemFingerprint,
		ServiceTier:       r.ServiceTier,
	}

	// Convert choices
	resp.Choices = lo.Map(r.Choices, func(c Choice, _ int) llm.Choice {
		return c.ToLLMChoice()
	})

	// Convert usage
	if r.Usage != nil {
		resp.Usage = r.Usage.ToLLMUsage()
	}

	// Convert error
	if r.Error != nil {
		resp.Error = &llm.ResponseError{
			StatusCode: r.Error.StatusCode,
			Detail:     r.Error.Detail,
		}
	}

	// Store citations in TransformerMetadata if present
	if len(r.Citations) > 0 {
		if resp.TransformerMetadata == nil {
			resp.TransformerMetadata = make(map[string]any)
		}

		resp.TransformerMetadata[TransformerMetadataKeyCitations] = r.Citations
	}

	return resp
}

// ToLLMChoice converts OpenAI Choice to unified llm.Choice.
func (c Choice) ToLLMChoice() llm.Choice {
	choice := llm.Choice{
		Index:        c.Index,
		FinishReason: c.FinishReason,
	}

	if c.Message != nil {
		msg := c.Message.ToLLMMessage()
		choice.Message = &msg
	}

	if c.Delta != nil {
		delta := c.Delta.ToLLMMessage()
		choice.Delta = &delta
	}

	if c.Logprobs != nil {
		choice.Logprobs = &llm.LogprobsContent{
			Content: lo.Map(c.Logprobs.Content, func(t TokenLogprob, _ int) llm.TokenLogprob {
				return llm.TokenLogprob{
					Token:   t.Token,
					Logprob: t.Logprob,
					Bytes:   t.Bytes,
					TopLogprobs: lo.Map(t.TopLogprobs, func(tl TopLogprob, _ int) llm.TopLogprob {
						return llm.TopLogprob{
							Token:   tl.Token,
							Logprob: tl.Logprob,
							Bytes:   tl.Bytes,
						}
					}),
				}
			}),
		}
	}

	return choice
}
