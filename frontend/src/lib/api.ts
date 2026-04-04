const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function resolveImageUrl(url: string): string {
  if (url.startsWith("/static/")) return `${API_BASE}${url}`;
  return url;
}

async function parseErrorResponse(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    return JSON.stringify(data);
  } catch {
    return await res.text().catch(() => "Unknown error");
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await parseErrorResponse(res);
    throw new Error(`API ${res.status}: ${error}`);
  }

  return res.json();
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  taste: {
    extract: async (data: FormData) => {
      const res = await fetch(`${API_BASE}/api/taste/extract`, {
        method: "POST",
        body: data,
      });

      if (!res.ok) {
        const error = await parseErrorResponse(res);
        throw new Error(`API ${res.status}: ${error}`);
      }

      return res.json();
    },

    update: (data: {
      user_id: string;
      taste_vector: number[];
      item_id: string;
      save_count: number;
    }) =>
      request<{
        taste_vector: number[];
        trend_fingerprint: Record<string, number>;
        aesthetic_attributes: Record<string, unknown>;
        price_tier: number[];
      }>("/api/taste/update", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },

  recommendations: {
    wardrobe: (data: {
      user_id: string;
      wardrobe_item_ids: string[];
      taste_vector: number[];
      taste_modes?: number[][];
      occasion_vectors?: Record<string, number[]>;
      trend_fingerprint?: Record<string, number>;
      anti_taste_vector?: number[];
      price_tier?: number[];
      aesthetic_label?: string;
      skipped_item_ids?: string[];
    }) =>
      request("/api/recommendations/wardrobe", {
        method: "POST",
        body: JSON.stringify(data),
      }),

    evaluateItem: (data: {
      product_url?: string;
      item_id?: string;
      user_id: string;
      wardrobe_item_ids: string[];
      taste_vector: number[];
    }) =>
      request("/api/recommendations/evaluate-item", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },

  catalog: {
    search: (params: { query?: string; slot?: string; page?: number; per_page?: number }) => {
      const searchParams = new URLSearchParams();
      if (params.query) searchParams.set("q", params.query);
      if (params.slot) searchParams.set("slot", params.slot);
      if (params.page) searchParams.set("page", String(params.page));
      if (params.per_page) searchParams.set("per_page", String(params.per_page));
      return request(`/api/catalog/search?${searchParams}`);
    },

    getItem: (id: string) => request(`/api/catalog/item/${id}`),

    tasteSearch: (data: {
      taste_vector: number[];
      slot?: string;
      top_k?: number;
      exclude_ids?: string[];
    }) =>
      request<{ items: Array<Record<string, unknown> & { taste_score: number }>; total: number }>(
        "/api/catalog/taste-search",
        { method: "POST", body: JSON.stringify(data) }
      ),
  },

  wardrobe: {
    get: (userId: string) => request(`/api/wardrobe/${userId}`),

    save: (data: { user_id: string; item_id: string }) =>
      request("/api/wardrobe/save", {
        method: "POST",
        body: JSON.stringify(data),
      }),

    remove: (saveId: string) =>
      request(`/api/wardrobe/save/${saveId}`, { method: "DELETE" }),
  },

  events: {
    log: (data: {
      user_id: string;
      event_type: string;
      module: string;
      item_id: string;
      score?: number;
      unlock_count?: number;
      taste_score?: number;
    }) =>
      request("/api/events/log", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },

  shopper: {
    plan: (data: {
      user_brief_json?: Record<string, unknown>;
      user_message?: string;
      taste_vector: number[];
      occasion_vectors?: Record<string, number[]>;
      wardrobe_item_ids?: string[];
      trend_fingerprint?: Record<string, number>;
      anti_taste_vector?: number[];
      price_tier?: number[];
    }) =>
      request<{
        plan: {
          occasion: string;
          slots_to_fill: string[];
          tone: string;
          max_price: number | null;
        };
        items: Record<string, unknown>[];
      }>("/api/shopper/plan", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },
};
