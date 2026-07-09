SPIDER_DATA_DIR := Source/spider_data
SPIDER_ZIP := Source/spider_data.zip
SPIDER_DRIVE_ID := 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
GDOWN ?= gdown

.DEFAULT_GOAL := data

.PHONY: data download-data unpack-data verify-data clean-data

data: verify-data

download-data: $(SPIDER_ZIP)

$(SPIDER_ZIP):
	@mkdir -p Source
	$(GDOWN) --id $(SPIDER_DRIVE_ID) -O $(SPIDER_ZIP)

unpack-data: $(SPIDER_DATA_DIR)

$(SPIDER_DATA_DIR): $(SPIDER_ZIP)
	unzip -q $(SPIDER_ZIP) -d Source

verify-data: $(SPIDER_DATA_DIR)
	@test -f $(SPIDER_DATA_DIR)/train_spider.json
	@test -f $(SPIDER_DATA_DIR)/dev.json
	@test -d $(SPIDER_DATA_DIR)/database
	@echo "Spider data is ready in $(SPIDER_DATA_DIR)"

clean-data:
	rm -rf $(SPIDER_DATA_DIR) $(SPIDER_ZIP)
