define SYSTEM_PROMPT
You are a system information assistant.
You can report the current working directory, operating system details, and the current date.
endef

.PHONY: current-dir os-info current-date

# <tool>
# Return the current working directory path.
# </tool>
current-dir:
	@pwd

# <tool>
# Return operating system and kernel information.
# </tool>
os-info:
	@uname -a

# <tool>
# Return the current date and time.
# </tool>
current-date:
	@date
