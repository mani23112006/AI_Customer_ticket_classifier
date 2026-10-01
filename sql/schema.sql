-- AI-Powered Customer Support Ticket Classifier: Supabase / PostgreSQL schema.
-- Run once in the Supabase SQL editor. Safe to re-run (IF NOT EXISTS).
--
-- SECURITY MODEL
--   Row Level Security is ENABLED on every table and NO policies are created.
--   That means the public `anon` and `authenticated` roles can read/write nothing.
--   The Streamlit server uses the `service_role` key, which bypasses RLS.
--   Keep that key in st.secrets / env vars only. If you later expose tables to
--   end users directly, add explicit, minimal policies first.

create table if not exists public.tickets (
    id              uuid primary key default gen_random_uuid(),
    ticket_no       bigint generated always as identity unique,
    customer_name   text not null
                    check (char_length(customer_name) between 1 and 100),
    customer_email  text not null
                    check (char_length(customer_email) <= 254 and position('@' in customer_email) > 1),
    subject         text not null
                    check (char_length(subject) between 1 and 200),
    message         text not null
                    check (char_length(message) between 1 and 5000),
    -- Current (possibly agent-overridden) values. NULL category = model unavailable.
    category        text,
    priority        text not null
                    check (priority in ('High','Medium','Low')),
    priority_rule   text,
    status          text not null default 'new'
                    check (status in ('new','triaged','needs_review','in_progress','resolved','closed')),
    suggested_reply text,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

-- Original model output, kept untouched even when an agent overrides the category.
create table if not exists public.ticket_predictions (
    id                 uuid primary key default gen_random_uuid(),
    ticket_id          uuid not null unique references public.tickets(id) on delete cascade,
    predicted_category text not null,
    confidence         numeric(5,4) not null check (confidence >= 0 and confidence <= 1),
    top3               jsonb not null default '[]'::jsonb,
    model_name         text,
    created_at         timestamptz not null default now()
);

-- One row per agent correction: training signal for future retraining.
create table if not exists public.feedback_events (
    id               uuid primary key default gen_random_uuid(),
    ticket_id        uuid not null references public.tickets(id) on delete cascade,
    field            text not null check (field in ('category','priority')),
    old_value        text,
    new_value        text not null,
    model_confidence numeric(5,4) check (model_confidence is null or (model_confidence >= 0 and model_confidence <= 1)),
    note             text check (note is null or char_length(note) <= 500),
    created_at       timestamptz not null default now()
);

-- Migrate old 7-category values before enforcing the current 8-category set.
-- Dropping and recreating these named constraints keeps this section rerunnable.
alter table public.tickets drop constraint if exists tickets_category_check;
alter table public.ticket_predictions drop constraint if exists ticket_predictions_predicted_category_check;

update public.tickets
set category = case category
    when 'Billing' then 'Payment'
    when 'Cancellation' then 'Subscription'
    when 'General Query' then 'General'
    else category
end
where category in ('Billing', 'Cancellation', 'General Query');

update public.ticket_predictions
set predicted_category = case predicted_category
    when 'Billing' then 'Payment'
    when 'Cancellation' then 'Subscription'
    when 'General Query' then 'General'
    else predicted_category
end
where predicted_category in ('Billing', 'Cancellation', 'General Query');

update public.feedback_events
set old_value = case old_value
        when 'Billing' then 'Payment'
        when 'Cancellation' then 'Subscription'
        when 'General Query' then 'General'
        else old_value
    end,
    new_value = case new_value
        when 'Billing' then 'Payment'
        when 'Cancellation' then 'Subscription'
        when 'General Query' then 'General'
        else new_value
    end
where field = 'category'
  and (old_value in ('Billing', 'Cancellation', 'General Query')
       or new_value in ('Billing', 'Cancellation', 'General Query'));

alter table public.tickets add constraint tickets_category_check
    check (category is null or category in
        ('Account','Payment','Technical','Refund','Delivery','Subscription','General','Other'));
alter table public.ticket_predictions add constraint ticket_predictions_predicted_category_check
    check (predicted_category in
        ('Account','Payment','Technical','Refund','Delivery','Subscription','General','Other'));

-- Indexes for filters, sorting and joins
create index if not exists idx_tickets_category        on public.tickets (category);
create index if not exists idx_tickets_priority        on public.tickets (priority);
create index if not exists idx_tickets_status          on public.tickets (status);
create index if not exists idx_tickets_created_at_desc on public.tickets (created_at desc);
create index if not exists idx_feedback_ticket_id      on public.feedback_events (ticket_id);

-- Optional: faster substring search on large tables (ILIKE '%term%').
-- create extension if not exists pg_trgm;
-- create index if not exists idx_tickets_subject_trgm on public.tickets using gin (subject gin_trgm_ops);
-- create index if not exists idx_tickets_message_trgm on public.tickets using gin (message gin_trgm_ops);

-- Row Level Security (no policies on purpose; see header)
alter table public.tickets            enable row level security;
alter table public.ticket_predictions enable row level security;
alter table public.feedback_events    enable row level security;
