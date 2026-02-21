CREATE OR REFRESH MATERIALIZED VIEW mv_vectorizable_metadata
AS
SELECT 
  -- Surrogate primary key
  CONCAT(table_catalog, '.', table_schema, '.', table_name) AS table_id,
  
  -- Keep identifiers for reference
  table_catalog,
  table_schema,
  table_name,
  
  -- Concatenated searchable text for vectorization
  CONCAT_WS(' | ',
    CONCAT('Catalog: ', table_catalog),
    CONCAT('Schema: ', table_schema),
    CONCAT('Table: ', table_catalog, '.', table_schema, '.', table_name),
    CONCAT('Type: ', COALESCE(table_type, '')),
    CONCAT('Format: ', COALESCE(data_source_format, '')),
    CONCAT('Table Owner: ', COALESCE(table_owner, '')),
    CONCAT('Table Comment: ', COALESCE(table_comment, '')),
    CONCAT('Table Tags: ', COALESCE(table_tags, '')),
    CONCAT('Schema Owner: ', COALESCE(schema_owner, '')),
    CONCAT('Schema Comment: ', COALESCE(schema_comment, '')),
    CONCAT('Schema Tags: ', COALESCE(schema_tags, '')),
    CONCAT('Catalog Owner: ', COALESCE(catalog_owner, '')),
    CONCAT('Catalog Comment: ', COALESCE(catalog_comment, '')),
    CONCAT('Catalog Tags: ', COALESCE(catalog_tags, ''))
  ) AS searchable_text,
  
  -- Individual fields for filtering/display
  table_type,
  data_source_format,
  table_comment,
  table_owner,
  table_tags,
  schema_comment,
  schema_tags,
  catalog_comment,
  catalog_tags
  
FROM mv_combined_metadata;
