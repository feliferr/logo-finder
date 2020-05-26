export BUCKET_NAME=rec-alg
export JOB_NAME="logo_classification_model_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/jobs/$JOB_NAME
export REGION=us-east1
export GCS_BASE_PATH="PATH_TO_GS"
export MODEL_BINARIES=$JOB_DIR


gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 2.1 \
    --python-version 3.7 \
    --scale-tier custom \
    --master-machine-type n1-standard-4 \
    --master-accelerator count=1,type=nvidia-tesla-k80 \
    --worker-count 1 \
    --worker-machine-type n1-standard-4 \
    --worker-accelerator count=1,type=nvidia-tesla-k80 \
    --module-name trainer.logo_classification_model \
    --package-path ./trainer \
    --region $REGION \
    -- \
    --train-base-path $GCS_BASE_PATH/logo-2k/train_and_test/train \
    --test-base-path $GCS_BASE_PATH/logo-2k/train_and_test/test \
    --train-image-meta $GCS_BASE_PATH/logo-2k/List/train_images_root.txt \
    --test-image-meta $GCS_BASE_PATH/logo-2k/List/test_images_root.txt \
    --classes-meta $GCS_BASE_PATH/logo-2k/List/Logo-2K+classes.txt 