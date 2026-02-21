CREATE OR REFRESH MATERIALIZED VIEW mv_tables_with_tags
AS
SELECT 
  tables.*,
  string_agg(CONCAT(tags.tag_name, '=', tags.tag_value), ', ') AS tags_collapsed
FROM system.information_schema.tables AS tables
LEFT JOIN system.information_schema.table_tags AS tags
  ON tags.catalog_name = tables.table_catalog
  AND tags.schema_name = tables.table_schema
  AND tags.table_name = tables.table_name
WHERE tables.table_catalog != 'system'
  AND tables.table_schema != 'information_schema'
GROUP BY 
  tables.table_catalog,
  tables.table_schema,
  tables.table_name,
  tables.table_type,
  tables.is_insertable_into,
  tables.commit_action,
  tables.comment,
  tables.created,
  tables.created_by,
  tables.last_altered,
  tables.last_altered_by,
  tables.table_owner,
  tables.data_source_format,
  tables.storage_sub_directory,
  tables.storage_path;
