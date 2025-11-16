with _views_raw as (
    select
        visit_id,
        user_id,
        (properties->'item_id')::int as item_id,
        REPLACE(
            LOWER(
                TRIM('"' FROM split_part(
                    split_part((properties->'referrer_url')::text, 'query=', 2),
                    '&', 1
                ))
            ),
            '+', ' '
        ) AS query,  -- TODO: remove leading/trailing whitespace
        CASE 
            WHEN TRIM('"' FROM split_part(
                split_part((properties->'referrer_url')::text, 'page=', 2),
                '&', 1
            )) = '' THEN 1
            ELSE CAST(TRIM('"' FROM split_part(
                split_part((properties->'referrer_url')::text, 'page=', 2),
                '&', 1
            )) AS int)
        END as page,
        (properties->'search_result_index')::int as raw_search_index,
        time
    from ahoy_events
    where name = 'Item viewed'
    order by time desc
),

views as (
    select 
        visit_id,
        user_id,
        query,
        item_id,
        items.name as item_name,
        time,
        CASE 
            WHEN page > 1 THEN raw_search_index + (page - 1) * 20
            ELSE raw_search_index
        END AS search_index
    from _views_raw
    inner join items
        on _views_raw.item_id = items.id
    where
        query is not null
        and query <> ''
),

hold_events as (
    select 
        events.visit_id,
        events.time,
        holds.id as hold_id,
        holds.member_id,
        items.id as item_id,
        items.name as item_name
    from ahoy_events events
    inner join holds
        on holds.id = (events.properties->'hold_id')::int
    inner join items
        on holds.item_id = items.id
    where events.name = 'Placed hold'
),

final as (
    select
        views.visit_id,
        views.query,
        views.item_id,
        views.item_name,
        -- make sure we only get (up to) one view and one hold for each item
        max(views.search_index) as search_index,
        min(views.time) as view_time,
        max(hold_events.hold_id) as hold_id
    from views
    left join hold_events
        on views.visit_id = hold_events.visit_id
        and views.item_id = hold_events.item_id
        and views.time < hold_events.time
    group by
        views.visit_id,
        views.query,
        views.item_id,
        views.item_name
)

select * from final
