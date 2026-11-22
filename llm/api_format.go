package llm

// IsOpenAIResponsesFormat reports whether format belongs to the OpenAI
// Responses protocol family, including its WebSocket and compact endpoints.
func IsOpenAIResponsesFormat(format APIFormat) bool {
	return format == APIFormatOpenAIResponse || format == APIFormatOpenAIResponseWebSocket || format == APIFormatOpenAIResponseCompact
}
