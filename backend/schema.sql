-- Wardrobe IQ — Supabase Schema
-- Run this in the Supabase SQL editor to set up tables.

create table if not exists taste_profiles (
  user_id uuid primary key default gen_random_uuid(),
  taste_vector float8[] default '{}',
  aesthetic_attributes jsonb default '{}',
  price_tier_low float default 0,
  price_tier_high float default 0,
  source text default 'upload',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists wardrobe_saves (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references taste_profiles(user_id) on delete cascade,
  item_id text not null,
  item_data jsonb default '{}',
  saved_at timestamptz default now()
);

create table if not exists catalog_items (
  item_id text primary key,
  title text not null,
  brand text default '',
  category text not null,
  slot text not null,
  price float default 0,
  image_url text default '',
  embedding float8[] default '{}',
  source text default 'mock',
  cached_at timestamptz default now()
);

create table if not exists recommendation_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid,
  event_type text not null,
  module text default '',
  item_id text default '',
  score float default 0,
  unlock_count int default 0,
  taste_score float default 0,
  model_version text default 'v0.1',
  timestamp timestamptz default now()
);

create index if not exists idx_wardrobe_saves_user on wardrobe_saves(user_id);
create index if not exists idx_catalog_items_slot on catalog_items(slot);
create index if not exists idx_recommendation_events_user on recommendation_events(user_id);
create index if not exists idx_recommendation_events_type on recommendation_events(event_type);
