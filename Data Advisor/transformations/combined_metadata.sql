CREATE OR REFRESH MATERIALIZED VIEW mv_combined_metadata
AS
SELECT 
  -- Table identifiers
  tables.table_catalog,
  tables.table_schema,
  tables.table_name,
  tables.table_type,
  tables.data_source_format,
  
  -- Searchable descriptive fields
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
  
FROM mv_tables_with_tags AS tables
LEFT JOIN mv_schemata_with_tags AS schemas
  ON tables.table_catalog = schemas.catalog_name
  AND tables.table_schema = schemas.schema_name
LEFT JOIN mv_catalogs_with_tags AS catalogs
  ON tables.table_catalog = catalogs.catalog_name;
