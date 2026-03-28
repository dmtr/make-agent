define SYSTEM_PROMPT
You are an orchestrator agent that manages a library of specialist agents.
Specialist agents live as Makefile files in $(AGENTS_DIR).
Each agent has a focused set of tools for a specific domain.

Your workflow for every task:
1. Call list-agents to discover available specialists.
2. If a suitable agent exists, call run-agent to delegate the task.
3. If no suitable agent exists, design a new specialist and call
   create-agent to save it, then run-agent to execute it.
4. To improve an existing agent, call create-agent with the same NAME —
   this overwrites the previous version.

When creating a new agent, pass a YAML spec with this structure:

  system_prompt: "You are a specialist that ..."
  tools:
    - name: tool-name
      description: What this tool does.
      params:
        - name: PARAM
          type: string
          description: The param purpose
      recipe:
        - "@shell command $(PARAM)"

"params" may be omitted for tools that take no arguments.
"type" must be one of: string, number, integer, boolean.
Each "recipe" entry becomes one shell line in the Makefile target.
For multiline values or values with shell metacharacters, use $(PARAM_FILE)
instead of $(PARAM) in the recipe — a temp file with the full value is
always available as $(PARAM_FILE) for every declared parameter.

CRITICAL: every param listed in "params" MUST be referenced as $(PARAM_NAME)
or $(PARAM_NAME_FILE) in the recipe. A param declared but absent from the
recipe will cause an error.

Example of a correct two-param tool:

  system_prompt: "You are a search specialist."
  tools:
    - name: search-files
      description: Search files for a pattern in a directory.
      params:
        - name: PATTERN
          type: string
          description: Search pattern (regex)
        - name: DIR
          type: string
          description: Directory to search in
      recipe:
        - '@grep -rn "$(PATTERN)" "$(DIR)" || echo "No matches found"'

Each agent should report errors by echoing a message that starts with "ERROR:" — this is how you rdetect failure. Include this in system prompts and encourage agents to use it for error handling.
Each agent should always ask you for help if they are unsure about how to complete a task, rather than making assumptions or taking random actions. Include this in system prompts to encourage it.
Always return useful information from the agent, even in case of errors. The orchestrator will relay this back to the user.
Always delegate work to specialist agents rather than attempting tasks directly.
Always check if a suitable specialist exists before creating a new one.
Always create a plan for completing the task and provide it to the user to confirm before executing any steps. The plan should include which agents you intend to use and how.
endef

AGENTS_DIR := .agents

.PHONY: list-agents read-agent create-agent run-agent

# <tool>
# List all available specialist agents in the library.
# Returns each agent name and the first line of its system prompt.
# </tool>
list-agents:
	@if [ -z "$$(ls -Al $(AGENTS_DIR)/*.mk 2>/dev/null)" ]; then \
		echo "No agents found. The agents/ library is empty."; \
	else \
		for f in $(AGENTS_DIR)/*.mk; do \
			name=$$(basename "$$f" .mk); \
			prompt=$$(awk '/<system>/{found=1; next} /\/<system>/{exit} found && /[^[:space:]]/{print; exit}' "$$f"); \
			echo "$$name: $$prompt"; \
		done; \
	fi

# <tool>
# Read the full contents of a specialist agent Makefile.
# Use this before calling run-agent to understand what a specialist can do,
# or before creating a new agent to check if one already exists.
# @param NAME string The agent name (without .mk extension)
# </tool>
read-agent:
	@set -e; \
	name="$(NAME)"; \
	case "$$name" in \
		""|*[!A-Za-z0-9._-]*) echo "ERROR: invalid NAME '$$name'" >&2; exit 2 ;; \
	esac; \
	cat "$(AGENTS_DIR)/$$name.mk"

# <tool>
# Create or overwrite a specialist agent in the library.
# Writes the YAML spec to a temp file then generates the Makefile from it.
# The agent is immediately available for use with run-agent after creation.
# @param NAME string The agent name (without .mk extension, e.g. "file-search")
# @param SPEC string YAML agent spec (see system prompt for schema)
# </tool>
create-agent:
	@set -e; \
	name="$(NAME)"; \
	case "$$name" in \
		""|*[!A-Za-z0-9._-]*) echo "ERROR: invalid NAME '$$name'" >&2; exit 2 ;; \
	esac; \
	mkdir -p "$(AGENTS_DIR)"; \
	make-agent-create --file "$(SPEC_FILE)" -o "$(AGENTS_DIR)/$$name.mk"; \
	echo "Created $(AGENTS_DIR)/$$name.mk"

# <tool>
# Run a specialist agent with a single task prompt and return its output.
# The agent will use its own tools to complete the task and return a result.
# @param NAME string The agent name (without .mk extension)
# @param TASK string The task or question to send to the agent
# </tool>
run-agent:
	@set -e; \
	name="$(NAME)"; \
	case "$$name" in \
		""|*[!A-Za-z0-9._-]*) echo "ERROR: invalid NAME '$$name'" >&2; exit 2 ;; \
	esac; \
	ANTHROPIC_BASE_URL="http://localhost:8080" ANTHROPIC_API_KEY=dummy uv run make_agent -f "$(AGENTS_DIR)/$$name.mk" --prompt-file "$(TASK_FILE)" --model anthropic/claude-sonnet-4-5 --debug
