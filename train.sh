#使用tensorflow object_detection api的流程
#下载最新tensorflow/models代码
#编译proto格式You need to download protoc version 3.3 (already compiled). Used protoc inside bin directory to run this command like this:
#mkdir protoc_3.3
#cd protoc_3.3
#tewget wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
#chmod 775 protoc-3.3.0-linux-x86_64.zip
#unzip protoc-3.3.0-linux-x86_64.zip
#cd ../models/
#/home/humayun/tensorflow/protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.
#设置环境
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim:`pwd`/object_detection

#下载对应的预训练好的模型文件model zoo：并编辑器对应的配置文件pipeline_config

#处理数据：将自己的文件格式写入tfrecord。

#运行
python model_main.py --alsologtostderr --model_dir=train_logs --pipeline_config_path=pipeline.config
