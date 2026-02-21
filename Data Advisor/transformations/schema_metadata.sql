CREATE OR REFRESH MATERIALIZED VIEW mv_schemata_with_tags
AS
SELECT 
  schemata.*,
  string_agg(CONCAT(tags.tag_name, '=', tags.tag_value), ', ') AS tags_collapsed
FROM system.information_schema.schemata AS schemata
LEFT JOIN system.information_schema.schema_tags AS tags
  ON schemata.catalog_name = tags.catalog_name
  AND schemata.schema_name = tags.schema_name
WHERE schemata.catalog_name != 'system'
  AND schemata.schema_name != 'information_schema'
GROUP BY 
  schemata.catalog_name,
  schemata.schema_name,
  schemata.schema_owner,
  schemata.comment,
  schemata.created,
  schemata.created_by,
  schemata.last_altered,
  schemata.last_altered_by;
