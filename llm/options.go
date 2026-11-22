package llm

type TransformOptions struct {
	// ArrayInstructions specifies whether the system instructions is an array.
	ArrayInstructions *bool `json:"array_instructions,omitempty"`

	// ArrayInputs specifies whether the inputs is an array.
	ArrayInputs *bool `json:"array_inputs,omitempty"`

	// DisableResponsesChatCompat disables the beta high-fidelity Responses-to-Chat
	// conversion for this request. The orchestrator stamps this from the selected
	// channel's EnableResponsesChatCompat option.
	DisableResponsesChatCompat bool `json:"disable_responses_chat_compat,omitempty"`

	// DefaultMaxTokens is the fallback max_tokens used by outbound transformers
	// that require the field (notably Anthropic Messages) when the client did not
	// set max_tokens or max_completion_tokens. Typically populated from
	// model_card.limit.output.
	DefaultMaxTokens *int64 `json:"default_max_tokens,omitempty"`
}
