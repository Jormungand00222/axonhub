package api

import (
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/objects"
)

func TestCodexImagesDefaultChannel(t *testing.T) {
	const imageURL = "data:image/png;base64,iVBORw0KGgo="
	upstream := responsesIntegrationServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := readRequestBody(r)
		assert.NoError(t, err)
		assert.Equal(t, "Bearer synthetic-key", r.Header.Get("Authorization"))
		assert.NotEmpty(t, r.Header.Get("X-Codex-Image-Turn-Id"))
		assert.Equal(t, "gpt-image-2", gjson.GetBytes(body, "model").String())
		assert.Equal(t, "auto", gjson.GetBytes(body, "quality").String())
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/images/generations":
			assert.Equal(t, "draw", gjson.GetBytes(body, "prompt").String())
			_, _ = io.WriteString(w, `{"created":123,"data":[{"b64_json":"Z2Vu"}]}`)
		case "/v1/images/edits":
			assert.Equal(t, imageURL, gjson.GetBytes(body, "images.0.image_url").String())
			assert.False(t, gjson.GetBytes(body, "response_format").Exists())
			_, _ = io.WriteString(w, `{"created":124,"data":[{"b64_json":"ZWRpdA=="}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	client, handlers, project := newResponsesIntegrationFixture(t, upstream.URL, func(create *ent.ChannelCreate) {
		create.SetType(channel.TypeCodex).SetSupportedModels([]string{"synthetic-model", "gpt-image-2"})
	})
	gateway := responsesIntegrationGateway(t, client, handlers, project)
	for _, tc := range []struct{ path, request, want string }{
		{"/v1/images/generations", `{"model":"gpt-image-2","prompt":"draw"}`, "Z2Vu"},
		{"/v1/images/edits", `{"model":"gpt-image-2","prompt":"edit","images":[{"image_url":"` + imageURL + `"}],"response_format":"b64_json"}`, "ZWRpdA=="},
	} {
		status, body := responsesIntegrationPost(t, gateway, tc.path, tc.request)
		require.Equal(t, http.StatusOK, status, string(body))
		require.Equal(t, tc.want, gjson.GetBytes(body, "data.0.b64_json").String())
	}
}

func TestCodexImageGenerationExplicitPassThrough(t *testing.T) {
	const body = `{"model":"synthetic-model","prompt":"draw","response_format":"b64_json","output_format":"webp","relay_option":"preserve"}`
	upstream := responsesIntegrationServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := readRequestBody(r)
		assert.NoError(t, err)
		assert.JSONEq(t, body, string(data))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"created":125,"data":[{"b64_json":"Z2Vu"}]}`)
	}))
	client, handlers, project := newResponsesIntegrationFixture(t, upstream.URL, func(create *ent.ChannelCreate) {
		create.SetType(channel.TypeCodex).SetSettings(&objects.ChannelSettings{PassThroughBody: new(true)})
	})
	gateway := responsesIntegrationGateway(t, client, handlers, project)
	status, response := responsesIntegrationPost(t, gateway, "/v1/images/generations", body)
	require.Equal(t, http.StatusOK, status, string(response))
	require.Equal(t, "Z2Vu", gjson.GetBytes(response, "data.0.b64_json").String())
}
