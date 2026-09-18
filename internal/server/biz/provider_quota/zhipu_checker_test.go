package provider_quota

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/ent/channel"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestZhipu_CheckQuota_CreditPlan(t *testing.T) {
	// Shape returned by the Coding Plan endpoint, with future reset timestamps.
	const body = `{"code":200,"msg":"Operation successful","data":{"limits":[
		{"type":"CREDIT_LIMIT","unit":3,"number":5,"usage":28000,"currentValue":7391,"remaining":20608,"percentage":26,"nextResetTime":4087947600000},
		{"type":"CREDIT_LIMIT","unit":6,"number":1,"usage":140000,"currentValue":19137,"remaining":120862,"percentage":13,"nextResetTime":4088534400000}
	],"level":"max"},"success":true}`
	for _, channelType := range []channel.Type{channel.TypeZhipu, channel.TypeZhipuAnthropic} {
		t.Run(string(channelType), func(t *testing.T) {
			client := httpclient.NewHttpClientWithClient(&http.Client{
				Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					require.Equal(t, http.MethodGet, req.Method)
					require.Equal(t, "https://open.bigmodel.cn/api/monitor/usage/quota/limit", req.URL.String())
					require.Equal(t, "test-key", req.Header.Get("Authorization"))
					return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}, nil
				}),
			})
			quota, err := NewZhipuQuotaChecker(client).CheckQuota(context.Background(), &ent.Channel{
				Type: channelType, Credentials: objects.ChannelCredentials{APIKey: "test-key"},
			})
			require.NoError(t, err)
			require.Equal(t, "zhipu", quota.ProviderType)
			require.Equal(t, "available", quota.Status)
			require.True(t, quota.Ready)
			require.Equal(t, "max", quota.RawData["level"])
			require.Len(t, quota.Limits, 2)
			for i, expected := range []struct {
				window string
				ratio  float64
				reset  int64
				period time.Duration
			}{
				{QuotaWindow5h, .26, 4087947600000, 5 * time.Hour},
				{QuotaWindowWeekly, .13, 4088534400000, 7 * 24 * time.Hour},
			} {
				limit := quota.Limits[i]
				require.Equal(t, expected.window, limit.Window)
				require.InDelta(t, expected.ratio, limit.UsageRatio, .0001)
				require.Equal(t, expected.reset, limit.NextResetAt.UnixMilli())
				require.Equal(t, expected.period, limit.NextResetAt.Sub(*limit.PeriodStart))
			}
			rows := quota.RawData["rows"].([]zhipuWindowRow)
			require.Equal(t, "five_hour", rows[0].Window)
			require.Equal(t, "weekly_limit", rows[1].Window)
		})
	}
}

func TestZhipuFamily_ExplicitPeriods(t *testing.T) {
	for _, provider := range []string{"zhipu", "zai"} {
		for _, limitType := range []string{"CREDIT_LIMIT", "TOKENS_LIMIT"} {
			t.Run(provider+"/"+limitType, func(t *testing.T) {
				// Weekly comes first and omits reset time; metadata must take
				// precedence over both ordering and the legacy reset heuristic.
				body := `{"success":true,"data":{"limits":[
					{"type":"LIMIT_TYPE","unit":6,"number":1,"percentage":100},
					{"type":"LIMIT_TYPE","unit":3,"number":5,"percentage":85,"nextResetTime":4088534400000},
					{"type":"TIME_LIMIT","percentage":100}
				]}}`
				quota, err := parseZhipuFamilyQuotaResponse([]byte(strings.ReplaceAll(body, "LIMIT_TYPE", limitType)), provider)
				require.NoError(t, err)
				require.Equal(t, provider, quota.ProviderType)
				require.Equal(t, "exhausted", quota.Status)
				require.False(t, quota.Ready)
				require.Len(t, quota.Limits, 2)
				require.Equal(t, QuotaWindow5h, quota.Limits[0].Window)
				require.Equal(t, "warning", quota.Limits[0].Status)
				require.Equal(t, QuotaWindowWeekly, quota.Limits[1].Window)
				require.Equal(t, "exhausted", quota.Limits[1].Status)
			})
		}
	}

	t.Run("weekly only", func(t *testing.T) {
		quota, err := parseZhipuQuotaResponse([]byte(`{"success":true,"data":{"limits":[{"type":"CREDIT_LIMIT","unit":6,"number":1,"percentage":13}]}}`))
		require.NoError(t, err)
		require.Len(t, quota.Limits, 1)
		require.Equal(t, QuotaWindowWeekly, quota.Limits[0].Window)
		require.Equal(t, "weekly_limit", quota.RawData["rows"].([]zhipuWindowRow)[0].Window)
	})

	t.Run("unknown period", func(t *testing.T) {
		_, err := parseZhipuQuotaResponse([]byte(`{"success":true,"data":{"limits":[{"type":"CREDIT_LIMIT","unit":99,"number":1,"percentage":13}]}}`))
		require.ErrorContains(t, err, "unsupported zhipu quota period")
	})
}

func TestZhipu_CheckQuota_HappyPath(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			require.Equal(t, http.MethodGet, req.Method)
			require.Equal(t, "https://open.bigmodel.cn/api/monitor/usage/quota/limit", req.URL.String())
			require.Equal(t, "test-key", req.Header.Get("Authorization"))
			require.Equal(t, "en-US,en", req.Header.Get("Accept-Language"))

			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 25.5, "nextResetTime": 1784322000000},
						{"type": "TOKENS_LIMIT", "percentage": 10.0, "nextResetTime": 1784476800000},
						{"type": "TIME_LIMIT", "percentage": 50.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		BaseURL:     "https://open.bigmodel.cn/api/paas/v4",
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	require.Equal(t, "available", quota.Status)
	require.True(t, quota.Ready)
	require.Equal(t, "zhipu", quota.ProviderType)
	require.Len(t, quota.Limits, 2)

	// First window (five_hour): 25.5% used
	require.InDelta(t, 0.255, quota.Limits[0].UsageRatio, 0.001)
	require.Equal(t, "available", quota.Limits[0].Status)

	// Second window (weekly_limit): 10.0% used
	require.InDelta(t, 0.10, quota.Limits[1].UsageRatio, 0.001)
	require.Equal(t, "available", quota.Limits[1].Status)

	rows, ok := quota.RawData["rows"].([]zhipuWindowRow)
	require.True(t, ok)
	require.Len(t, rows, 2)
	require.Equal(t, "five_hour", rows[0].Window)
	require.Equal(t, "weekly_limit", rows[1].Window)
	require.Equal(t, "standard", quota.RawData["level"])
}

func TestZhipu_CheckQuota_SingleWindow(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "basic",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 5.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	require.Len(t, quota.Limits, 1)
	require.Equal(t, "available", quota.Limits[0].Status)
	require.InDelta(t, 0.05, quota.Limits[0].UsageRatio, 0.001)
}

func TestZhipu_CheckQuota_Warning(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 90.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	require.Equal(t, "warning", quota.Status)
	require.True(t, quota.Ready)
}

func TestZhipu_CheckQuota_Exhausted(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 100.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	require.Equal(t, "exhausted", quota.Status)
	require.False(t, quota.Ready)
}

func TestZhipu_CheckQuota_APIError(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": false,
				"code": 500,
				"msg": "当前用户不存在coding plan"
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	_, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "当前用户不存在coding plan")
}

func TestZhipu_CheckQuota_MissingAPIKey(t *testing.T) {
	checker := NewZhipuQuotaChecker(nil)
	_, err := checker.CheckQuota(context.Background(), &ent.Channel{Type: channel.TypeZhipu})
	require.ErrorContains(t, err, "channel has no API key")
}

func TestZhipu_CheckQuota_APIKeysFallback(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			require.Equal(t, "fallback-key", req.Header.Get("Authorization"))

			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 15.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type: channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{
			APIKey:  "",
			APIKeys: []string{"fallback-key"},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "available", quota.Status)
}

func TestZhipu_CheckQuota_APIKeysFallbackSkipsBlankEntries(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			require.Equal(t, "fallback-key", req.Header.Get("Authorization"))

			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 15.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type: channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{
			APIKey:  "",
			APIKeys: []string{"", "  ", "fallback-key"},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "available", quota.Status)
}

func TestZhipu_CheckQuota_MalformedJSON(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(`not json`)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	_, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "failed to parse zhipu quota response")
}

func TestZhipu_CheckQuota_NoData(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{"success": true, "code": 200}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	_, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "no data")
}

func TestZhipu_CheckQuota_NoTokensLimit(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TIME_LIMIT", "percentage": 50.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	_, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "no TOKENS_LIMIT")
}

func TestZhipu_CheckQuota_NoResetIsFiveHour(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			// Weekly returned first but has a reset time; the entry without reset
			// must be the 5-hour bucket regardless of API order.
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 96.0, "nextResetTime": 1784541710987},
						{"type": "TOKENS_LIMIT", "percentage": 0.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	require.Len(t, quota.Limits, 2)

	rows, ok := quota.RawData["rows"].([]zhipuWindowRow)
	require.True(t, ok)
	// The no-reset entry is five_hour, even though it came second in the response.
	require.Equal(t, "five_hour", rows[0].Window)
	require.InDelta(t, 0.0, rows[0].UsedPercent, 0.001)
	require.Nil(t, rows[0].ResetAt)
	require.Equal(t, "weekly_limit", rows[1].Window)
	require.InDelta(t, 96.0, rows[1].UsedPercent, 0.001)
	require.NotNil(t, rows[1].ResetAt)
}

func TestZhipu_CheckQuota_BothHaveResetTime(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			// Both windows have reset time — mirrors real API when 5h has usage.
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 1.0, "nextResetTime": 1784384374404},
						{"type": "TOKENS_LIMIT", "percentage": 96.0, "nextResetTime": 1784541710987}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)

	rows, ok := quota.RawData["rows"].([]zhipuWindowRow)
	require.True(t, ok)
	// First entry is five_hour, both have reset times.
	require.Equal(t, "five_hour", rows[0].Window)
	require.InDelta(t, 1.0, rows[0].UsedPercent, 0.001)
	require.NotNil(t, rows[0].ResetAt)
	require.Equal(t, "weekly_limit", rows[1].Window)
	require.InDelta(t, 96.0, rows[1].UsedPercent, 0.001)
	require.NotNil(t, rows[1].ResetAt)
}

func TestZhipu_SupportsChannel(t *testing.T) {
	checker := NewZhipuQuotaChecker(nil)

	require.True(t, checker.SupportsChannel(&ent.Channel{Type: channel.TypeZhipu}))
	require.True(t, checker.SupportsChannel(&ent.Channel{Type: channel.TypeZhipuAnthropic}))
	require.False(t, checker.SupportsChannel(&ent.Channel{Type: channel.TypeZai}))
	require.False(t, checker.SupportsChannel(&ent.Channel{Type: channel.TypeZaiAnthropic}))
	require.False(t, checker.SupportsChannel(&ent.Channel{Type: channel.TypeOpenai}))
}

func TestBuildZhipuQuotaURL(t *testing.T) {
	require.Equal(t, "https://open.bigmodel.cn/api/monitor/usage/quota/limit", buildZhipuQuotaURL("https://open.bigmodel.cn/api/paas/v4"))
	require.Equal(t, "https://open.bigmodel.cn/api/monitor/usage/quota/limit", buildZhipuQuotaURL("https://open.bigmodel.cn/api/anthropic"))
	require.Equal(t, "https://open.bigmodel.cn/api/monitor/usage/quota/limit", buildZhipuQuotaURL(""))
	require.Equal(t, "https://open.bigmodel.cn/api/monitor/usage/quota/limit", buildZhipuQuotaURL("not a URL"))
}

func TestZhipu_CheckQuota_WithResetTime(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 50.0, "nextResetTime": 4087947600000},
						{"type": "TOKENS_LIMIT", "percentage": 30.0, "nextResetTime": 4088534400000}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	require.NotNil(t, quota.NextResetAt)
	// Earlier reset time
	require.Equal(t, int64(4087947600), quota.NextResetAt.Unix())
}

func TestZhipu_CheckQuota_OverallStatusWorst(t *testing.T) {
	httpClient := httpclient.NewHttpClientWithClient(&http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			body := `{
				"success": true,
				"code": 200,
				"data": {
					"level": "standard",
					"limits": [
						{"type": "TOKENS_LIMIT", "percentage": 50.0},
						{"type": "TOKENS_LIMIT", "percentage": 95.0}
					]
				}
			}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		}),
	})

	checker := NewZhipuQuotaChecker(httpClient)
	quota, err := checker.CheckQuota(context.Background(), &ent.Channel{
		Type:        channel.TypeZhipu,
		Credentials: objects.ChannelCredentials{APIKey: "test-key"},
	})
	require.NoError(t, err)
	// five_hour=available, weekly=warning → overall=warning
	require.Equal(t, "warning", quota.Status)
}
