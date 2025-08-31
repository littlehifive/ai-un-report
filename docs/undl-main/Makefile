QUERY := "Evaluation of women and peace and security in field-based missions : elections and political transitions : report of the Office of Internal Oversight Services"
LONG_QUERY := "Women in peacekeeping"
QUERY_OUTPUT := downloads/output_query.json
LONG_QUERY_OUTPUT := downloads/output_long_query.json

ID := 515307
ID_OUTPUT := downloads/output_id.json

.PHONY: help clean query long-query id

clean:
	rm -rf downloads/*
	@echo "Cleaned downloads folder"

help:
	poetry run python main.py --help

query:
	poetry run python main.py 	\
		-q ${QUERY} 				\
		-o ${QUERY_OUTPUT} 			\
		-v

long-query:
	poetry run python main.py 	\
		-q ${LONG_QUERY} 			\
		-o ${LONG_QUERY_OUTPUT} 	\
		-v

id:
	poetry run python main.py 	\
		--id ${ID} 				\
		-o ${ID_OUTPUT} 			\
		-v

test-long:
	poetry run python undl/test.py -p $(LONG_QUERY)

test:
	poetry run pytest -v
