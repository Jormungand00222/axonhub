package llm

import (
	"encoding/json"
	"slices"
)

// ProviderExtensions carries provider/API-format private data that should not
// be serialized through the common llm request/response JSON model.
type ProviderExtensions struct {
	OpenAIResponses *OpenAIResponsesProviderExtensions `json:"-"`
}

type OpenAIResponsesProviderExtensions struct {
	Request *OpenAIResponsesRequestExtensions `json:"-"`
}

type OpenAIResponsesRequestExtensions struct {
	ReasoningContext        string                       `json:"-"`
	RawFields               map[string]json.RawMessage   `json:"-"`
	RawTools                []OpenAIResponsesRawFragment `json:"-"`
	ToolSignatures          []string                     `json:"-"`
	RawToolChoice           json.RawMessage              `json:"-"`
	RawInputItems           []OpenAIResponsesRawFragment `json:"-"`
	OmittedInputItemIndices []int                        `json:"-"`
	RawInputMessages        []string                     `json:"-"`
	RawInputTools           []string                     `json:"-"`
}

type OpenAIResponsesRawFragment struct {
	Type          string `json:"-"`
	Name          string `json:"-"`
	CallID        string `json:"-"`
	OriginalIndex int    `json:"-"`
	// RepresentedToolCount is the number of structured tools replaced when Raw is replayed.
	RepresentedToolCount int             `json:"-"`
	Raw                  json.RawMessage `json:"-"`
}

func EnsureOpenAIResponsesProviderExtensions(req *Request) *OpenAIResponsesProviderExtensions {
	if req == nil {
		return nil
	}

	if req.ProviderExtensions == nil {
		req.ProviderExtensions = &ProviderExtensions{}
	}

	if req.ProviderExtensions.OpenAIResponses == nil {
		req.ProviderExtensions.OpenAIResponses = &OpenAIResponsesProviderExtensions{}
	}

	return req.ProviderExtensions.OpenAIResponses
}

func CloneProviderExtensions(src *ProviderExtensions) *ProviderExtensions {
	if src == nil {
		return nil
	}

	cloned := &ProviderExtensions{}
	if src.OpenAIResponses != nil {
		cloned.OpenAIResponses = &OpenAIResponsesProviderExtensions{}
		if src.OpenAIResponses.Request != nil {
			cloned.OpenAIResponses.Request = &OpenAIResponsesRequestExtensions{
				ReasoningContext:        src.OpenAIResponses.Request.ReasoningContext,
				ToolSignatures:          slices.Clone(src.OpenAIResponses.Request.ToolSignatures),
				OmittedInputItemIndices: slices.Clone(src.OpenAIResponses.Request.OmittedInputItemIndices),
				RawFields:               cloneRawMessageMap(src.OpenAIResponses.Request.RawFields),
				RawTools:                cloneOpenAIResponsesRawFragments(src.OpenAIResponses.Request.RawTools),
				RawToolChoice:           cloneRawMessage(src.OpenAIResponses.Request.RawToolChoice),
				RawInputItems:           cloneOpenAIResponsesRawFragments(src.OpenAIResponses.Request.RawInputItems),
				RawInputMessages:        append([]string(nil), src.OpenAIResponses.Request.RawInputMessages...),
				RawInputTools:           append([]string(nil), src.OpenAIResponses.Request.RawInputTools...),
			}
		}
	}

	return cloned
}

func cloneRawMessageMap(src map[string]json.RawMessage) map[string]json.RawMessage {
	if len(src) == 0 {
		return nil
	}

	out := make(map[string]json.RawMessage, len(src))
	for key, value := range src {
		out[key] = cloneRawMessage(value)
	}

	return out
}

func cloneOpenAIResponsesRawFragments(src []OpenAIResponsesRawFragment) []OpenAIResponsesRawFragment {
	if len(src) == 0 {
		return nil
	}

	out := make([]OpenAIResponsesRawFragment, len(src))
	for i := range src {
		out[i] = src[i]
		out[i].Raw = cloneRawMessage(src[i].Raw)
	}

	return out
}

func cloneRawMessage(src json.RawMessage) json.RawMessage {
	if len(src) == 0 {
		return nil
	}

	return append(json.RawMessage(nil), src...)
}

// NamespaceMappingMetadataKey records the original Responses namespace by flat function name.
const NamespaceMappingMetadataKey = "namespace_mapping"

// NamespaceForFunctionName accepts both the original map and its JSON-decoded form.
func NamespaceForFunctionName(metadata map[string]any, name string) string {
	switch mapping := metadata[NamespaceMappingMetadataKey].(type) {
	case map[string]string:
		return mapping[name]
	case map[string]any:
		namespace, _ := mapping[name].(string)
		return namespace
	default:
		return ""
	}
}
