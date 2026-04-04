id: string | int                  # unique point id for qdrant
vector: float[2048]

payload:
  source_file_id: string          # stable file-level ID
  source_path: string             # full path in trusted dir
  file_name: string
  extension: string
  mime_type: string

  modality: string                # text | image | ocr_text | transcript_text | audio
  pipeline_name: string           # which pipeline produced this embedding

  chunk_id: string                # unique ID for this embedding item
  chunk_index: int | null         # for chunked text/transcripts
  embedding_family: string        # optional grouping label, e.g. "primary_text", "ocr", "raw_audio"

  extracted_text: string | null   # transcript / OCR / text chunk when applicable
  content_hash: string            # hash of source file contents
  parent_content_hash: string | null   # optional if chunk-level hash differs

  created_at: string              # ISO timestamp
  updated_at: string              # ISO timestamp

  source_type: string | null      # optional, e.g. meeting, notes, email, image
  org_id: string | null           # optional for later family/group retrieval
  metadata: object | null         # extra modality-specific metadata