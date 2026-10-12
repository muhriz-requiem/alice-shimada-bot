# Setting Up a New Character Bot

This guide walks through creating a new character bot (e.g., "Bob") using the AliceShimadaBot codebase. The architecture is designed so that adding a new character requires **only data files** — no Python code changes.

## Prerequisites

- Python 3.11+
- Discord bot application created on [Discord Developer Portal](https://discord.com/developers/applications)
- Groq API key (or Shapes/Ollama backend)
- Redis instance (for multi-bot/shared state)

---

## 1. Create Character Data Folder

Create a new folder under `character/data/` with your character's name (lowercase):

```
character/data/bob/
├── identity.json          # Required: display name, identity keywords, core facts
├── lore.txt               # Required: character backstory
├── examples.json          # Required: style examples (fast + full)
├── gifs.json              # Required: GIF URLs by emotion/intent
├── intents.json           # Required: keyword triggers for intents
├── relationship_tones.json # Required: relationship definitions
├── prompt_rules.json      # Required: prompt shaping rules
└── boko_gifts.json        # Optional: gift categories (if using /boko-gift)
```

### Required Files Explained

#### `identity.json`
```json
{
  "display_name": "Bob",
  "identity_keywords": ["bob", "creator", "father"],
  "core_facts": {
    "Favorite Food": "Pizza is Bob's favorite food. He always knows this.",
    "Pet": "Bob has a pet cat named Whiskers."
  }
}
```

| Field | Purpose |
|-------|---------|
| `display_name` | Character's name shown in prompts |
| `identity_keywords` | Words that trigger identity/emotion boosts |
| `core_facts` | ABSOLUTE-TRUTH facts the model must never forget |

#### `lore.txt`
Free-form markdown/text backstory. Used directly in system prompt.

#### `examples.json`
```json
{
  "fast": [
    "User: \"Hi\" -> Bob: \"H-Hey there.\"",
    "User: \"Bye\" -> Bob: \"S-See you later.\""
  ],
  "full": [
    "User: \"Hello Bob\" -> Bob: \"H-Hello! How can I help?\"",
    "User (Father): \"Good job son\" -> Bob: \"T-Thanks, Dad! That means a lot.\""
  ]
}
```

- `fast`: Used for short messages (greetings, acknowledgments)
- `full`: Used for normal conversations
- Tagged examples: `User (Role):` — only used when speaking to that role

#### `gifs.json`
```json
{
  "happy": ["https://tenor.com/view/...gif"],
  "sad": ["https://tenor.com/view/...gif"],
  "shocked": ["https://tenor.com/view/...gif"],
  "agree": ["https://tenor.com/view/...gif"],
  "agree_special": ["https://tenor.com/view/...gif"],
  "goodnight": ["https://tenor.com/view/...gif"]
}
```
Tenor/GIPHY URLs. Keys match emotion names + intent names.

#### `intents.json`
```json
{
  "goodnight": ["goodnight", "gn", "sleep well"],
  "agree": ["agree", "you're right", "true"],
  "agree_special": ["i think so too", "probably"]
}
```
Keyword lists for intent detection. Keys must match GIF keys.

#### `relationship_tones.json`
```json
{
  "stranger": {"tone": "polite", "trust": 0.2, "warmth": 0.2, "description": "..."},
  "familiar": {"tone": "friendly", "trust": 0.5, "warmth": 0.5, "description": "..."},
  "trusted": {"tone": "warm", "trust": 0.8, "warmth": 0.8, "description": "..."},
  "guardian-figure": {"tone": "reverent", "trust": 0.95, "warmth": 0.9, "description": "..."},
  "anonymous": {"tone": "gentle", "trust": 0.3, "warmth": 0.5, "description": "..."}
}
```
Defines how the character behaves toward different relationship tiers.

#### `prompt_rules.json`
```json
{
  "identity_trigger_phrases": ["who am i", "what is my name"],
  "memory_blocked_phrases": ["student at", "roleplay", "faction"],
  "never_say_phrases": ["How can I help you?", "As an AI", "I am an assistant"]
}
```
- `identity_trigger_phrases`: Triggers strict identity mode
- `memory_blocked_phrases`: Memories containing these are filtered out
- `never_say_phrases`: Assistant-speak the character must avoid

#### `boko_gifts.json` (Optional)
```json
{
  "categories": {
    "food": {
      "label": "Food (pizza, snacks, drinks)",
      "placeholder": "A slice of pepperoni pizza...",
      "reaction_hint": "Bob's eyes light up at the smell."
    }
  },
  "fallback_reaction": "He accepts it with a grin."
}
```
Only needed if using `/boko-gift` command.

---

## 2. Configure Environment Variables

Copy `.env.example` to `.env` and set the following with your bot's prefix (e.g., `BOB_`):

```bash
# Discord
BOB_DISCORD_TOKEN=your-bob-discord-token
BOB_OWNER_IDS=your-discord-user-id
BOB_DEV_GUILD_ID=your-test-guild-id  # optional

# AI Backend
BOB_AI_BACKEND=groq
BOB_GROQ_API_KEY=your-groq-key
BOB_GROQ_MODEL=openai/gpt-oss-120b

# Character
BOB_CHARACTER_NAME=bob

# Redis (for shared state)
BOB_REDIS_HOST=localhost
BOB_REDIS_PORT=6379

# Health checks
BOB_HEALTH_PORT=8081  # Different from Alice's 8080

# Logging
BOB_LOG_JSON=true
```

**Key point**: Each bot gets its own prefixed env vars. The codebase reads them via `BOT_NAME` prefix automatically.

---

## 3. Set Up Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create new application → "Bot" section → "Reset Token"
3. Enable **Message Content Intent** and **Server Members Intent**
4. Copy token to `BOB_DISCORD_TOKEN`
5. Invite bot to your server with `applications.commands` scope

---

## 4. Set Up Redis (Required for Multi-Bot)

```bash
# Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Or install locally
# Ubuntu: sudo apt install redis-server
# macOS: brew install redis
```

Set `BOB_REDIS_HOST` and `BOB_REDIS_PASSWORD` if using auth.

---

## 5. Run the Bot

```bash
# Install deps (once per machine)
pip install -r requirements.txt

# Run Bob
BOT_NAME=BOB python main.py
```

The `BOT_NAME=BOB` prefix tells the config system to look for `BOB_*` env vars and use `data/BOB/` for data storage.

---

## 6. Verify It Works

1. **Health checks**:
   - `curl http://localhost:8081/healthz` → `{"status":"alive"}`
   - `curl http://localhost:8081/ready` → `{"status":"ready"}`
   - `curl http://localhost:8081/metrics` → Prometheus metrics

2. **Discord commands**:
   - `/chat message:Hello` → Bob replies in character
   - `/bobpost` → Random post from Bob's subreddit
   - `/admin-stats` (owner only) → Shows Bob's stats

---

## 7. Optional: Customize Prompt Template

The system prompt uses `templates/system_prompt.j2`. To customize per-character:

1. Copy `templates/system_prompt.j2` to `character/data/bob/system_prompt.j2`
2. Modify as needed
3. Update `prompt_builder.py` to load character-specific template:

```python
# In prompt_builder.py, add:
def _get_template(character: Character):
    char_template = Path(__file__).parent.parent / "data" / character.key / "system_prompt.j2"
    if char_template.exists():
        return _JINJA_ENV.get_template(f"../data/{character.key}/system_prompt.j2")
    return _JINJA_ENV.get_template("system_prompt.j2")
```

---

## 8. Deploying Multiple Bots

### Option A: Same Host, Different Ports
```bash
# Terminal 1
BOT_NAME=ALICE python main.py

# Terminal 2
BOT_NAME=BOB python main.py
```

### Option B: Docker Compose
```yaml
# docker-compose.yml
services:
  alice:
    build: .
    environment:
      - BOT_NAME=ALICE
      - ALICE_DISCORD_TOKEN=${ALICE_DISCORD_TOKEN}
      - ALICE_GROQ_API_KEY=${GROQ_API_KEY}
      - ALICE_REDIS_HOST=redis
    ports:
      - "8080:8080"
    depends_on: [redis]

  bob:
    build: .
    environment:
      - BOT_NAME=BOB
      - BOB_DISCORD_TOKEN=${BOB_DISCORD_TOKEN}
      - BOB_GROQ_API_KEY=${GROQ_API_KEY}
      - BOB_REDIS_HOST=redis
    ports:
      - "8081:8081"
    depends_on: [redis]

  redis:
    image: redis:7-alpine
```

---

## 9. Common Customizations

### Change AI Backend
```bash
# Ollama (local)
BOB_AI_BACKEND=ollama
BOB_OLLAMA_HOST=http://localhost:11434
BOB_OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct

# Shapes (when available)
BOB_AI_BACKEND=shapes
BOB_SHAPESINC_API_KEY=...
BOB_SHAPESINC_SHAPE_USERNAME=...
```

### Add New Subreddit
1. Add to `config.py`:
```python
"NewSubreddit": {
    "raw_dir": REDDIT_DATA_DIR / "new_raw",
    "posts_file": REDDIT_DATA_DIR / "new_posts.json",
    "display_name": "r/NewSubreddit",
    "embed_color": 0xFF00FF,
}
```
2. Add command in `cogs/reddit_cog.py`:
```python
@app_commands.command(name="newpost", description="Random post from r/NewSubreddit")
async def newpost(self, interaction: discord.Interaction) -> None:
    await self._send_random_post(interaction, "NewSubreddit")
```

### Disable Redis (Single Bot)
```bash
# Just don't set REDIS_HOST or set to empty
BOB_REDIS_HOST=
```
The bot falls back to in-memory state automatically.

---

## 10. File Checklist for New Character

- [ ] `character/data/<name>/identity.json`
- [ ] `character/data/<name>/lore.txt`
- [ ] `character/data/<name>/examples.json`
- [ ] `character/data/<name>/gifs.json`
- [ ] `character/data/<name>/intents.json`
- [ ] `character/data/<name>/relationship_tones.json`
- [ ] `character/data/<name>/prompt_rules.json`
- [ ] `character/data/<name>/boko_gifts.json` (optional)
- [ ] `.env` with `BOT_NAME=<NAME>` and prefixed variables
- [ ] Discord bot created and invited
- [ ] Redis running (if multi-bot)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Unknown subreddit key" | Check `config.REDDIT_SUBREDDITS` matches command |
| "GROQ_API_KEY not set" | Verify prefixed env var (`BOB_GROQ_API_KEY`) |
| "Redis connection failed" | Check `REDIS_HOST`/`PORT`, ensure Redis running |
| Character doesn't recognize identity | Check `social/data/identities.json` and `identity_keywords` |
| Tests fail | Run `python -m pytest tests/ -v` for details |

---

## Summary

The entire character personality lives in **data files** under `character/data/<name>/`. Adding a new bot = new data folder + prefixed `.env` + `BOT_NAME` prefix. No Python changes needed unless you want custom prompt templates or new commands.