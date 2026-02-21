CREATE OR REFRESH MATERIALIZED VIEW mv_combined_metadata_column
AS
SELECT 
  -- Surrogate key
  CONCAT(columns.table_catalog, '.', columns.table_schema, '.', columns.table_name, '.', columns.column_name) AS column_key,
  
  -- Column identifiers
  columns.table_catalog,
  columns.table_schema,
  columns.table_name,
  columns.column_name,
  columns.ordinal_position,
  
  -- Column properties
  columns.data_type,
  columns.full_data_type,
  columns.is_nullable,
  columns.column_default,
  columns.comment AS column_comment,
  columns.tags_collapsed AS column_tags,
  columns.partition_index,
  columns.is_updatable,
  
  -- Numeric type details
  columns.numeric_precision,
  columns.numeric_scale,
  
  -- Table context
  tables.table_type,
  tables.data_source_format,
  tables.comment AS table_comment,
  tables.table_owner,
  tables.tags_collapsed AS table_tags,
  
  -- Schema context
  schemas.schema_owner,
  schemas.comment AS schema_comment,
  schemas.tags_collapsed AS schema_tags,
  
  -- Catalog context
  catalogs.catalog_owner,
  catalogs.comment AS catalog_comment,
  catalogs.tags_collapsed AS catalog_tags
  
FROM mv_columns_with_tags AS columns
LEFT JOIN mv_tables_with_tags AS tables
  ON columns.table_catalog = tables.table_catalog
  AND columns.table_schema = tables.table_schema
  AND columns.table_name = tables.table_name
LEFT JOIN mv_schemata_with_tags AS schemas
  ON columns.table_catalog = schemas.catalog_name
  AND columns.table_schema = schemas.schema_name
LEFT JOIN mv_catalogs_with_tags AS catalogs
  ON columns.table_catalog = catalogs.catalog_name;
