CREATE OR REFRESH MATERIALIZED VIEW mv_columns_with_tags
AS
SELECT 
  columns.*,
  string_agg(CONCAT(tags.tag_name, '=', tags.tag_value), ', ') AS tags_collapsed
FROM system.information_schema.columns AS columns
LEFT JOIN system.information_schema.column_tags AS tags
  ON tags.catalog_name = columns.table_catalog
  AND tags.schema_name = columns.table_schema
  AND tags.table_name = columns.table_name
  AND tags.column_name = columns.column_name
WHERE columns.table_catalog != 'system'
  AND columns.table_schema != 'information_schema'
GROUP BY 
  columns.table_catalog,
  columns.table_schema,
  columns.table_name,
  columns.column_name,
  columns.ordinal_position,
  columns.column_default,
  columns.is_nullable,
  columns.full_data_type,
  columns.data_type,
  columns.character_maximum_length,
  columns.character_octet_length,
  columns.numeric_precision,
  columns.numeric_precision_radix,
  columns.numeric_scale,
  columns.datetime_precision,
  columns.interval_type,
  columns.interval_precision,
  columns.maximum_cardinality,
  columns.is_identity,
  columns.identity_generation,
  columns.identity_start,
  columns.identity_increment,
  columns.identity_maximum,
  columns.identity_minimum,
  columns.identity_cycle,
  columns.is_generated,
  columns.generation_expression,
  columns.is_system_time_period_start,
  columns.is_system_time_period_end,
  columns.system_time_period_timestamp_generation,
  columns.is_updatable,
  columns.partition_index,
  columns.comment;
