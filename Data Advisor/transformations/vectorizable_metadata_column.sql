CREATE OR REFRESH MATERIALIZED VIEW mv_vectorizable_metadata_column
AS
SELECT 
  -- Surrogate primary key
  column_key,
  
  -- Keep identifiers for reference
  table_catalog,
  table_schema,
  table_name,
  column_name,
  ordinal_position,
  
  -- Concatenated searchable text for vectorization
  CONCAT_WS(' | ',
    CONCAT('Catalog: ', table_catalog),
    CONCAT('Schema: ', table_schema),
    CONCAT('Table: ', table_catalog, '.', table_schema, '.', table_name),
    CONCAT('Column: ', column_name),
    CONCAT('Full Column Path: ', table_catalog, '.', table_schema, '.', table_name, '.', column_name),
    CONCAT('Data Type: ', COALESCE(data_type, '')),
    CONCAT('Full Data Type: ', COALESCE(full_data_type, '')),
    CONCAT('Nullable: ', COALESCE(is_nullable, '')),
    CONCAT('Column Comment: ', COALESCE(column_comment, '')),
    CONCAT('Column Tags: ', COALESCE(column_tags, '')),
    CONCAT('Partition Index: ', COALESCE(CAST(partition_index AS STRING), 'Not Partitioned')),
    CONCAT('Table Type: ', COALESCE(table_type, '')),
    CONCAT('Table Format: ', COALESCE(data_source_format, '')),
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
  
  -- Individual column fields for filtering/display
  data_type,
  full_data_type,
  is_nullable,
  column_default,
  column_comment,
  column_tags,
  partition_index,
  is_updatable,
  numeric_precision,
  numeric_scale,
  
  -- Table context fields
  table_type,
  data_source_format,
  table_comment,
  table_owner,
  table_tags,
  
  -- Schema context fields
  schema_owner,
  schema_comment,
  schema_tags,
  
  -- Catalog context fields
  catalog_owner,
  catalog_comment,
  catalog_tags
  
FROM mv_combined_metadata_column;
