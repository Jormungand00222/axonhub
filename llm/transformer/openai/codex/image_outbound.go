package codex

import (
	"cmp"
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

type ImageParams struct {
	TokenProvider oauth.TokenGetter
	BaseURL       string
	EndpointPath  string
}

type imageOutbound struct {
	transformer.Outbound
	tokens   oauth.TokenGetter
	official bool
}

func (t *imageOutbound) AllowPassThroughBody(_ context.Context, req *llm.Request, _ *httpclient.Request) bool {
	return req.APIFormat != llm.APIFormatOpenAIImageEdit
}

func NewImageOutboundTransformer(params ImageParams) (transformer.Outbound, error) {
	if params.TokenProvider == nil {
		return nil, fmt.Errorf("token provider is required")
	}
	baseURL := params.BaseURL
	if baseURL == "" || baseURL == "https://api.openai.com/v1" {
		baseURL = codexBaseURL
	}
	baseURL = strings.Replace(strings.Replace(baseURL, "wss://", "https://", 1), "ws://", "http://", 1)
	out, err := openai.NewOutboundTransformerWithConfig(&openai.Config{
		PlatformType: openai.PlatformOpenAI,
		BaseURL:      baseURL, EndpointPath: params.EndpointPath, ImageEditJSON: true,
		APIKeyProvider: auth.NewStaticKeyProvider(""),
	})
	if err != nil {
		return nil, err
	}
	return &imageOutbound{Outbound: out, tokens: params.TokenProvider, official: isOfficialCodexBaseURL(baseURL)}, nil
}

func (t *imageOutbound) TransformRequest(ctx context.Context, req *llm.Request) (*httpclient.Request, error) {
	if req == nil || req.RequestType != llm.RequestTypeImage || req.Image == nil {
		return nil, fmt.Errorf("%w: native Images endpoint requires an image request", transformer.ErrInvalidRequest)
	}
	if req.Stream != nil && *req.Stream {
		return nil, fmt.Errorf("%w: image request does not support streaming", transformer.ErrInvalidRequest)
	}
	creds, err := t.tokens.Get(ctx)
	if err != nil {
		return nil, err
	}
	// The Codex Images endpoints expect concrete defaults on translated requests.
	// Leave the caller's request intact for retries through a different channel.
	image := *req.Image
	image.Prompt = strings.TrimSpace(image.Prompt)
	image.Background = cmp.Or(strings.TrimSpace(image.Background), "auto")
	image.Quality = cmp.Or(strings.TrimSpace(image.Quality), "auto")
	image.Size = cmp.Or(strings.TrimSpace(image.Size), "auto")
	imageReq := *req
	imageReq.Image = &image
	imageReq.Model = strings.TrimSpace(req.Model)
	if image.Prompt == "" || imageReq.Model == "" {
		return nil, fmt.Errorf("%w: Codex image requests require a model and prompt", transformer.ErrInvalidRequest)
	}
	if t.official && req.APIFormat == llm.APIFormatOpenAIImageGeneration {
		// Official Codex uses a narrower generation schema than OpenAI-compatible relays.
		image = llm.ImageRequest{
			Prompt: image.Prompt, N: image.N, Background: image.Background,
			Quality: image.Quality, Size: image.Size,
		}
	}
	if req.APIFormat == llm.APIFormatOpenAIImageEdit {
		// Legacy clients send these OpenAI options even though the direct
		// Codex edit endpoint does not consume them.
		image.ResponseFormat = ""
		image.Style = ""
		image.PartialImages = nil
	}
	hreq, err := t.Outbound.TransformRequest(ctx, &imageReq)
	if err != nil {
		return nil, err
	}
	hreq.Auth = &httpclient.AuthConfig{Type: httpclient.AuthTypeBearer, APIKey: creds.AccessToken}
	var raw http.Header
	if req.RawRequest != nil {
		raw = req.RawRequest.Headers
	}
	// Keep Responses negotiation and stale body encodings off the JSON Images
	// request, including during the pipeline's subsequent inbound-header merge.
	raw.Del(ResponsesLiteHeader)
	raw.Del(BetaFeaturesHeader)
	raw.Del("Content-Encoding")
	raw.Del("X-Codex-Image-Turn-Id")
	ScrubClientFingerprintHeaders(raw)
	hreq.Headers.Set("X-Codex-Image-Turn-Id", uuid.NewString())
	hreq.SkipInboundQueryMerge = true
	hreq.Headers.Set("User-Agent", codexDefaultUserAgent)
	hreq.Headers.Set("Originator", AxonHubOriginator)
	for _, header := range append([]string{"Originator"}, PassthroughHeaders...) {
		if value := raw.Get(header); value != "" {
			hreq.Headers.Set(header, value)
		}
	}
	session := raw.Get(SessionHeaderHyphen)
	if session == "" {
		session = raw.Get(SessionHeader)
	}
	if session == "" {
		session = ExtractSessionIDFromTurnMetadata(raw.Get(TurnMetadataHeader))
	}
	if session == "" {
		session, _ = shared.GetSessionID(ctx)
	}
	if session == "" {
		session = uuid.NewString()
	}
	raw.Del(SessionHeader)
	hreq.Headers.Set(SessionHeaderHyphen, session)
	applyCodexSessionHeaders(hreq.Headers, ExtractChatGPTAccountIDFromJWT(creds.AccessToken))
	return hreq, nil
}

func (t *imageOutbound) TransformResponse(ctx context.Context, raw *httpclient.Response) (*llm.Response, error) {
	resp, err := t.Outbound.TransformResponse(ctx, raw)
	if err != nil {
		return nil, err
	}
	if resp.Image == nil || len(resp.Image.Data) == 0 {
		return nil, fmt.Errorf("%w: Codex image response contains no image data", transformer.ErrInvalidResponse)
	}
	resp.Object = "image.generation"
	resp.APIFormat = llm.APIFormat(raw.Request.APIFormat)
	return resp, nil
}
