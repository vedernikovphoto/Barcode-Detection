train:
	export PYTHONPATH=$(PWD) && python src/train.py config/config.yaml

install:
	pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

inference:
	export PYTHONPATH=$(PWD) && python yolov5/detect.py \
		--weights "$(PWD)/model_weights/best_model_yolov5.onnx" \
		--source "$(PWD)/inference_images" \
		--project "$(PWD)/output_images" \
		--name "results" \
		--img-size 640 \
		--conf-thres 0.25 \
		--iou-thres 0.45 \
		--save-txt \
		--save-conf \
		--device cpu

download:
	@echo "Fetching download URL..."
	@wget -q -O - 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/pRFNuxLQUZcDDg' \
	| grep -oP '(?<="href":")[^"]*' > download_url.txt
	@echo "Downloading dataset..."
	@wget -O dataset.zip -i download_url.txt
	@rm download_url.txt
	@echo "Download complete."
	@echo "Extracting files..."
	python -c "import zipfile; zipfile.ZipFile('dataset.zip').extractall('.')"
	rm dataset.zip
