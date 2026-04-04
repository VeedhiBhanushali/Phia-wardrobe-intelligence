-- Wardrobe IQ: analytics events (impression, click, save, dismiss, skip).
--
-- Apply once in Supabase Dashboard:
--   SQL Editor → New query → paste this file → Run
--
-- After running, wait a few seconds for PostgREST to refresh its schema cache,
-- or reload the project from Settings → API if inserts still 404.

create table if not exists public.wardrobe_events (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  event_type text not null,
  module text not null,
  item_id text not null,
  score double precision,
  unlock_count integer,
  taste_score double precision,
  model_version text not null default 'v0.2',
  created_at timestamptz not null default now()
);

create index if not exists idx_wardrobe_events_user_id
  on public.wardrobe_events (user_id);

create index if not exists idx_wardrobe_events_created_at
  on public.wardrobe_events (created_at desc);

create index if not exists idx_wardrobe_events_user_created
  on public.wardrobe_events (user_id, created_at desc);

comment on table public.wardrobe_events is 'Wardrobe IQ event log; inserts from FastAPI only.';

alter table public.wardrobe_events enable row level security;

-- Backend uses the project secret key (sb_secret_* or legacy service_role JWT);
-- that role bypasses RLS. Do not use the anon/publishable key for inserts.
