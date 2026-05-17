define DESCRIPTION
Read, write, and edit text files on the filesystem
endef

.PHONY: list-files count-lines read-file read-lines write-file append-to-file replace-in-file

list-files:
	@ls -la "$$DIR"

count-lines:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@wc -l < "$$FILE"

read-file:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@cat "$$FILE"

read-lines:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@if [ "$$END" = "0" ] || [ -z "$$END" ]; then \
		tail -n "+$$START" "$$FILE"; \
	else \
		awk "NR>=$$START && NR<=$$END" "$$FILE"; \
	fi

write-file:
	@mkdir -p "$$(dirname $$FILE)"
	@printf '%s' "$$CONTENT" > "$$FILE" && echo "Written: $$FILE"

append-to-file:
	@printf '%s' "$$CONTENT" >> "$$FILE" && echo "Appended to: $$FILE"

replace-in-file:
	@[ -f "$$FILE" ] || { echo "Error: file not found: $$FILE"; exit 1; }
	@python3 -c "\
import sys; \
p, o, n = sys.argv[1:]; \
t = open(p).read(); \
(print('Error: text not found in ' + p), sys.exit(1)) if o not in t \
else (open(p, 'w').write(t.replace(o, n, 1)), print('Replaced in: ' + p)) \
" "$$FILE" "$$OLD" "$$NEW"
