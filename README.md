# ATLAS Course Assistant

ATLAS is a course-specific AI teaching assistant for Northeastern University. It answers from uploaded syllabi and lecture transcripts, displays the supporting source and a short excerpt, and says plainly when the course materials do not contain an answer.

The application supports two course types:

- Legacy courses stored in `knowledge/` and listed on the public landing page.
- Faculty pilot courses created in the web dashboard and stored on a persistent volume. These courses use private student links and each professor's own Anthropic or OpenAI API key.

## Five-professor Arlington pilot

The pilot is invitation-only and allows five active or invited professors. Its workflow is:

1. The administrator signs in at `/pilot-admin` and creates a one-time faculty invitation.
2. The professor opens the invitation, creates an account, and adds their own Anthropic key, OpenAI key, or both.
3. The professor creates a course, selects its provider and model, uploads a syllabus and lecture transcripts, and generates its concept map.
4. The professor publishes the course and copies its private student link into Canvas or an email.
5. ATLAS tracks the course's monthly question and token totals without storing student questions or generated answers.

Faculty keys are encrypted at rest. The dashboard only reveals whether each key is present and its last four characters. The application never shares one professor's keys, courses, or documents with another professor.

Private student links are deliberately unguessable, but they are bearer links: anyone who receives a link can use that course's monthly allowance. Keep the link inside the intended Canvas course. A monthly per-course question limit prevents unbounded use during the pilot.

## Local development

### 1. Install and configure

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Generate an encryption key once:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set these values in `.env` for local pilot development:

```dotenv
ATLAS_PILOT_ENABLED=true
ATLAS_DATA_DIR=./data/pilot
ATLAS_ENCRYPTION_KEY=the-generated-fernet-key
ATLAS_ADMIN_PASSWORD=a-long-unique-administrator-password
ATLAS_MAX_PILOT_PROFESSORS=5
ATLAS_SECURE_COOKIES=false
```

Use `ATLAS_SECURE_COOKIES=false` only over local HTTP. Production must keep it `true`.

### 2. Run and test

```bash
uvicorn main:app --reload --port 8000
python -m unittest discover -s tests -v
```

Open `http://localhost:8000/pilot-admin` to create the first invitation.

## Railway deployment

Pilot data must not live on Railway's ephemeral application filesystem. Before enabling the pilot:

1. Attach a persistent Railway volume to the `web` service and mount it at `/data`.
2. Keep one application replica while ATLAS uses SQLite.
3. Set `ATLAS_DATA_DIR=/data/atlas`.
4. Set `ATLAS_PILOT_ENABLED=true`, `ATLAS_MAX_PILOT_PROFESSORS=5`, and `ATLAS_SECURE_COOKIES=true`.
5. Set `ATLAS_PUBLIC_BASE_URL` to the service's public HTTPS origin, without a trailing slash.
6. Set strong secret values for `ATLAS_ENCRYPTION_KEY` and `ATLAS_ADMIN_PASSWORD`.
7. Keep the encryption key stable. Changing it makes saved professor API keys unreadable.
8. Back up the volume, including `atlas_pilot.sqlite3` and the `courses/` directory.

The Railway start command is:

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

The application fails startup when the pilot is enabled without its persistent data path or encryption key. When the pilot is disabled, existing public courses continue to run normally.

## Provider and model choices

Each pilot course stores its own provider and model choice. A professor may change either choice while a course is published. The new choice applies to the next student question; a request already in progress finishes with the previous choice. Existing materials, the private student link, and the current concept map do not change. The professor can regenerate the concept map separately if desired.

ATLAS only permits a course to use Anthropic when that professor has saved an Anthropic key, and only permits OpenAI when that professor has saved an OpenAI key. Keys are not interchangeable. Usage totals are separated by provider and model, and model changes are recorded. ATLAS never changes a course's provider or model automatically.

The faculty dashboard presents a short list of current general-purpose models from each provider. Existing pilot courses remain on Claude Sonnet 4.6 until their professor changes them. New Anthropic courses default to Claude Sonnet 5, and new OpenAI courses default to GPT-5.6 Terra.

## Course documents and privacy

Accepted file types are `.docx`, `.md`, `.pdf`, `.pptx`, and `.txt`. Each file is limited to 25 MB and one million extracted characters. Each course is limited to 100 documents, 250 MB, and five million extracted characters. A professor can create up to ten pilot courses. Readable text is extracted when the professor uploads the file.

For pilot courses, ATLAS stores:

- professor account details and a password hash;
- encrypted Anthropic and OpenAI keys, when supplied, and their last four characters;
- course metadata and uploaded course files;
- monthly question and token counts;
- thumbs-up or thumbs-down feedback and an optional comment.

ATLAS does not store the student question or generated answer for pilot analytics.

Concept-map generation is an explicit faculty action. It makes one setup request through the provider and model selected for that course, records the provider, model, and token totals for cost visibility, and does not reduce the course's student question allowance.

## Canvas distribution

The current pilot uses a Canvas External URL:

1. Copy the published student link from the faculty dashboard.
2. In the Canvas module, select **Add item**, then **External URL**.
3. Paste the link and select **Load in a new tab**.

A full Canvas LTI integration with roster-based access and single sign-on is a separate phase. The private-link workflow avoids requiring students to create another account during the initial pilot.

## Legacy course materials

Legacy courses are registered in `knowledge/courses.json`. Each course directory may contain:

```text
knowledge/<course-id>/
├── syllabus.md
├── transcripts/
└── concept_map.json
```

The legacy administrator endpoints use `ATLAS_ADMIN_PASSWORD`; there is no hard-coded administrator key in the application.
