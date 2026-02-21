CREATE OR REFRESH MATERIALIZED VIEW mv_catalogs_with_tags
AS
SELECT 
  catalogs.*,
  string_agg(CONCAT(tags.tag_name, '=', tags.tag_value), ', ') AS tags_collapsed
FROM system.information_schema.catalogs
LEFT JOIN guanjie_catalog.information_schema.catalog_tags AS tags
  ON catalogs.catalog_name = tags.catalog_name
WHERE catalogs.catalog_name != 'system'
GROUP BY 
  catalogs.catalog_name,
  catalogs.catalog_owner,
  catalogs.comment,
  catalogs.created,
  catalogs.created_by,
  catalogs.last_altered,
  catalogs.last_altered_by;
