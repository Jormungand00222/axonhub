package codex

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"net/url"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/oauth"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type failingTokenGetter struct{}

func (failingTokenGetter) Get(context.Context) (*oauth.OAuthCredentials, error) {
	return nil, errors.New("refresh failed")
}

func TestCodexImageEditJSON(t *testing.T) {
	// Given: the official JSON image-reference schema, including file IDs and a mask.
	raw := &httpclient.Request{Headers: http.Header{"Content-Type": {"application/json"}, "Session-Id": {"image-session"}}, Body: []byte(`{"model":"gpt-image-2","prompt":"make blue","images":[{"image_url":"https://example.com/image.png"},{"file_id":"file-fixture"}],"mask":{"file_id":"mask-fixture"},"n":2,"quality":"high","input_fidelity":"high","size":"1024x1024","user":"fixture-user","output_format":"webp","output_compression":0,"moderation":"low","background":"transparent"}`)}
	in := openai.NewImageEditInboundTransformer()
	out, err := NewImageOutboundTransformer(ImageParams{BaseURL: "https://relay.example/backend#", TokenProvider: oauth.NewAPIKeyTokenProvider(auth.NewStaticKeyProvider("image-token").Get)})
	require.NoError(t, err)
	// When: a public JSON edit is translated to the Codex outbound.
	req, err := in.TransformRequest(t.Context(), raw)
	require.NoError(t, err)
	hreq, err := out.TransformRequest(t.Context(), req)
	// Then: all machine-consumed fields survive, with no Responses tool payload.
	require.NoError(t, err)
	require.Equal(t, "https://relay.example/backend/images/edits", hreq.URL)
	require.Equal(t, "application/json", hreq.Headers.Get("Content-Type"))
	require.Equal(t, "application/json", hreq.Headers.Get("Accept"))
	require.Equal(t, "image-token", hreq.Auth.APIKey)
	require.Equal(t, "image-session", hreq.Headers.Get("Session-Id"))
	var got, want map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(hreq.Body, &got))
	require.NoError(t, json.Unmarshal(raw.Body, &want))
	require.Equal(t, want, got)
	require.Equal(t, llm.APIFormatOpenAIImageEdit.String(), hreq.APIFormat)
}

func TestCodexImageEditJSONInvalidReferences(t *testing.T) {
	for _, body := range []string{
		`{"prompt":"edit","images":[{}]}`, `{"prompt":"edit","images":[{"image_url":"https://example.com/a.png","file_id":"file-a"}]}`,
		`{"prompt":"edit","images":[{"image_url":" "}]}`, `{"prompt":"edit","images":[{"file_id":"file-a","unknown":1}]}`,
		`{"prompt":"edit","images":[]}`, `{"prompt":"edit","images":[{"file_id":"file-a"}],"mask":{}}`,
	} {
		t.Run(body, func(t *testing.T) {
			// Given / When
			_, err := openai.NewImageEditInboundTransformer().TransformRequest(t.Context(), &httpclient.Request{Headers: http.Header{"Content-Type": {"application/json"}}, Body: []byte(body)})
			// Then
			require.Error(t, err)
		})
	}
}

func TestCodexImagesRejectStreamingBeforeTokenRefresh(t *testing.T) {
	out, err := NewImageOutboundTransformer(ImageParams{TokenProvider: failingTokenGetter{}, BaseURL: "https://relay.example/v1"})
	require.NoError(t, err)
	stream := true
	_, err = out.TransformRequest(t.Context(), &llm.Request{
		Model: "gpt-image-2", RequestType: llm.RequestTypeImage, APIFormat: llm.APIFormatOpenAIImageGeneration,
		Stream: &stream, Image: &llm.ImageRequest{Prompt: "draw"},
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, transformer.ErrInvalidRequest)
	assert.NotContains(t, err.Error(), "refresh failed")
}

func TestCodexImageMultipartJSONDriverCapturesURLAuthImageMaskAnd403(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":{"message":"forbidden"}}`))
	}))
	defer server.Close()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	require.NoError(t, form.WriteField("model", "gpt-image-2"))
	require.NoError(t, form.WriteField("prompt", "make blue"))
	imagePart, err := form.CreatePart(textproto.MIMEHeader{"Content-Disposition": {`form-data; name="image"; filename="input.png"`}, "Content-Type": {"image/png"}})
	require.NoError(t, err)
	_, err = imagePart.Write([]byte("\x89PNG image"))
	require.NoError(t, err)
	maskPart, err := form.CreatePart(textproto.MIMEHeader{"Content-Disposition": {`form-data; name="mask"; filename="mask.png"`}, "Content-Type": {"image/png"}})
	require.NoError(t, err)
	_, err = maskPart.Write([]byte("\x89PNG mask"))
	require.NoError(t, err)
	require.NoError(t, form.Close())

	in := openai.NewImageEditInboundTransformer()
	llmReq, err := in.TransformRequest(t.Context(), &httpclient.Request{Method: http.MethodPost, Headers: http.Header{"Content-Type": {form.FormDataContentType()}}, Body: body.Bytes()})
	require.NoError(t, err)
	out, err := NewImageOutboundTransformer(ImageParams{TokenProvider: oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "image-token"}), BaseURL: server.URL})
	require.NoError(t, err)
	hreq, err := out.TransformRequest(t.Context(), llmReq)
	require.NoError(t, err)
	httpReq, err := http.NewRequestWithContext(t.Context(), hreq.Method, hreq.URL, bytes.NewReader(hreq.Body))
	require.NoError(t, err)
	for key, values := range hreq.Headers {
		for _, value := range values {
			httpReq.Header.Add(key, value)
		}
	}
	httpReq.Header.Set("Authorization", "Bearer "+hreq.Auth.APIKey)
	resp, err := http.DefaultClient.Do(httpReq)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusForbidden, resp.StatusCode)
	require.Equal(t, "/v1/images/edits", gotPath)
	require.Equal(t, "Bearer image-token", gotAuth)
	require.Contains(t, string(gotBody), `"images":[{"image_url":"data:image/png;base64,`)
	require.Contains(t, string(gotBody), `"mask":{"image_url":"data:image/png;base64,`)
}

func TestCodexDirectImagesRoundTrip(t *testing.T) {
	const reference = "data:image/png;base64,iVBORw0KGgo="
	const result = `{"created":123,"data":[{"b64_json":"aW1hZ2U=","generation_id":"img-1"}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}`
	for _, tc := range []struct {
		name                 string
		in                   *openai.ImageInboundTransformer
		body, path, wantBody string
	}{
		{
			name:     "generate",
			in:       openai.NewImageGenerationInboundTransformer(),
			body:     `{"model":"gpt-image-2","prompt":"draw a tree","n":2}`,
			path:     "/images/generations",
			wantBody: `{"model":"gpt-image-2","prompt":"draw a tree","n":2,"background":"auto","quality":"auto","size":"auto"}`,
		},
		{
			name:     "edit",
			in:       openai.NewImageEditInboundTransformer(),
			body:     `{"model":"gpt-image-2","prompt":"edit a tree","quality":"high","background":"transparent","response_format":"b64_json","images":[{"image_url":" ` + reference + ` "}]}`,
			path:     "/images/edits",
			wantBody: `{"model":"gpt-image-2","prompt":"edit a tree","quality":"high","background":"transparent","size":"auto","images":[{"image_url":"` + reference + `"}]}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, http.MethodPost, r.Method)
				assert.Equal(t, tc.path, r.URL.Path)
				assert.Empty(t, r.URL.RawQuery)
				assert.Equal(t, "Bearer image-token", r.Header.Get("Authorization"))
				assert.Equal(t, "application/json", r.Header.Get("Accept"))
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
				assert.Equal(t, "image-session", r.Header.Get(SessionHeaderHyphen))
				assert.Equal(t, "image-session", r.Header.Get(ThreadIDHeader))
				assert.Equal(t, "image-session", ExtractSessionIDFromTurnMetadata(r.Header.Get(TurnMetadataHeader)))
				assert.Equal(t, codexDefaultVersion, r.Header.Get("Version"))
				assert.Equal(t, codexDefaultUserAgent, r.UserAgent())
				assert.NotEmpty(t, r.Header.Get("X-Codex-Image-Turn-Id"))
				assert.NotEqual(t, "stale", r.Header.Get("X-Codex-Image-Turn-Id"))
				assert.Empty(t, r.Header.Get(ResponsesLiteHeader))
				assert.Empty(t, r.Header.Get(BetaFeaturesHeader))
				assert.Empty(t, r.Header.Get("Content-Encoding"))
				body, err := io.ReadAll(r.Body)
				assert.NoError(t, err)
				assert.JSONEq(t, tc.wantBody, string(body))
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, result)
			}))
			defer server.Close()
			out, err := NewImageOutboundTransformer(ImageParams{
				BaseURL:       server.URL + "#",
				TokenProvider: oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "image-token"}),
			})
			require.NoError(t, err)
			pipe := pipeline.NewFactory(httpclient.NewHttpClient()).Pipeline(tc.in, out)
			res, err := pipe.Process(t.Context(), &httpclient.Request{
				Method: http.MethodPost, Body: []byte(tc.body),
				Query: url.Values{"local": {"true"}},
				Headers: http.Header{
					"Content-Type": {"application/json"}, "Content-Encoding": {"identity"},
					SessionHeaderHyphen: {"image-session"}, ResponsesLiteHeader: {"true"},
					BetaFeaturesHeader: {"responses"}, "X-Codex-Image-Turn-Id": {"stale"},
					"User-Agent": {"inbound-test"},
				},
			})
			require.NoError(t, err)
			require.False(t, res.Stream)
			require.JSONEq(t, result, string(res.Response.Body))
		})
	}
}

func TestCodexDirectImageResponseRequiresData(t *testing.T) {
	out, err := NewImageOutboundTransformer(ImageParams{TokenProvider: failingTokenGetter{}})
	require.NoError(t, err)
	_, err = out.TransformResponse(t.Context(), &httpclient.Response{
		Request:    &httpclient.Request{APIFormat: llm.APIFormatOpenAIImageGeneration.String()},
		StatusCode: http.StatusOK, Body: []byte(`{"created":123,"data":[]}`),
	})
	require.ErrorIs(t, err, transformer.ErrInvalidResponse)
}

func TestCodexImageGenerationBodyByUpstream(t *testing.T) {
	const body = `{"model":"gpt-image-2","prompt":"draw","n":2,"response_format":"b64_json","output_format":"webp","background":"transparent","quality":"low","size":"1024x1024"}`
	const officialBody = `{"model":"gpt-image-2","prompt":"draw","n":2,"background":"transparent","quality":"low","size":"1024x1024"}`
	for _, tc := range []struct{ name, baseURL, want string }{
		{"official default", "", officialBody},
		{"legacy base URL", "https://api.openai.com/v1", officialBody},
		{"compatible relay", "https://relay.example/codex#", body},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := NewImageOutboundTransformer(ImageParams{
				BaseURL:       tc.baseURL,
				TokenProvider: oauth.NewStaticTokenProvider(&oauth.OAuthCredentials{AccessToken: "test-token"}),
			})
			require.NoError(t, err)
			req, err := openai.NewImageGenerationInboundTransformer().TransformRequest(t.Context(), &httpclient.Request{
				Headers: http.Header{"Content-Type": {"application/json"}}, Body: []byte(body),
			})
			require.NoError(t, err)
			hreq, err := out.TransformRequest(t.Context(), req)
			require.NoError(t, err)
			require.JSONEq(t, tc.want, string(hreq.Body))
			require.True(t, out.(transformer.PassThroughBodyPolicy).AllowPassThroughBody(t.Context(), req, hreq))
			require.JSONEq(t, body, string(req.RawRequest.Body))
		})
	}
}
