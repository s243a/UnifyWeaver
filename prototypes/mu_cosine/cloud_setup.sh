#!/usr/bin/env bash
# TEMPLATE — environment setup script for the μ-cosine ML work (Claude-Code-on-the-web).
# Paste/adapt this into your environment's SETUP-SCRIPT field. It contains NO secrets: the Dropbox
# credentials are read from env vars you set yourself in the env-vars field.
#
# Requires a NETWORK POLICY that allows:
#   - huggingface.co            (MiniLM weights, via sentence-transformers)
#   - api.dropboxapi.com, content.dropboxapi.com   (rclone, only if you use the storage block)
#
# CREDENTIALS / SECURITY:
#   - Set DROPBOX_APP_KEY / DROPBOX_APP_SECRET / DROPBOX_REFRESH_TOKEN in the env-vars field yourself.
#   - Create the Dropbox app as **Scoped access — App folder** so the token physically cannot reach
#     anything outside /Apps/<YourApp>/. That is the REAL boundary; an rclone path prefix is not —
#     any command could still type `dropbox:` and reach a full-Dropbox app's entire account.
#   - The env-vars field is visible to anyone who can edit the environment and there is no secrets
#     store, so treat the refresh token as exposable; app-folder scoping is the safety net.
#   - Generate the refresh token once via `rclone config` on your OWN machine (it runs the OAuth flow
#     in a browser); copy the refresh_token out. An agent must NOT enter credentials for you.
set -e

# 1. Python deps for training the encoder. (The stdlib proof/scaffold/generator need none of this.)
pip install -r prototypes/mu_cosine/requirements.txt

# 2. OPTIONAL — rclone for large artifacts that outgrow git (MiniLM cache, model checkpoints,
#    full-graph embeddings ~1.5 GB, scored label sets), persisted to a Dropbox APP FOLDER.
if [ -n "${DROPBOX_REFRESH_TOKEN:-}" ]; then
  curl -fsSL https://rclone.org/install.sh | bash || true
  mkdir -p ~/.config/rclone
  cat > ~/.config/rclone/rclone.conf <<EOF
[dropbox]
type = dropbox
client_id = ${DROPBOX_APP_KEY}
client_secret = ${DROPBOX_APP_SECRET}
token = {"access_token":"","token_type":"bearer","refresh_token":"${DROPBOX_REFRESH_TOKEN}","expiry":"2000-01-01T00:00:00Z"}
EOF
  chmod 600 ~/.config/rclone/rclone.conf
  # Pre-pull stable artifacts here so they bake into the session snapshot (mind the ~30 GB disk).
  # Paths are relative to the app folder, e.g. dropbox:cache == /Apps/<YourApp>/cache.
  mkdir -p prototypes/mu_cosine/cache
  rclone copy "dropbox:cache" prototypes/mu_cosine/cache --transfers 4 || true
fi

# During a session:
#   rclone copy "dropbox:datasets" ./data --progress        # download
#   rclone copy ./outputs "dropbox:outputs" --progress      # upload results (copy never deletes)
#   rclone ls   "dropbox:"                                   # list (app-folder root)
# Use `rclone sync` only when you genuinely want an exact mirror (it deletes).
