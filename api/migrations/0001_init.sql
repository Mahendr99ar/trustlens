-- Only scores are stored: no review text, no user data.
CREATE TABLE IF NOT EXISTS reviews (
  asin          TEXT    NOT NULL,
  review_id     TEXT    NOT NULL,
  rating        REAL    NOT NULL,
  ai_prob       REAL    NOT NULL,
  aspects       TEXT    NOT NULL DEFAULT '{}',
  threshold     REAL    NOT NULL,
  model_version TEXT    NOT NULL,
  install_id    TEXT    NOT NULL,
  updated_at    INTEGER NOT NULL,
  PRIMARY KEY (asin, review_id)
);
CREATE INDEX IF NOT EXISTS idx_reviews_asin ON reviews (asin);
