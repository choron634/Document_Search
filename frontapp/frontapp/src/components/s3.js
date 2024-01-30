// AWS SDKのインポート
import AWS from 'aws-sdk';

// AWSの設定
AWS.config.update({
  region: 'ap-southeast-2', // あなたのリージョンを設定
  accessKeyId: 'AKIAZQBRFUXI2BJV6GZX', // あなたのアクセスキーを設定
  secretAccessKey: 'JBTxyFsZhsB2HldVIPes0cN41DqyoOhyTjy4rHSY' // あなたのシークレットキーを設定
});

// S3のインスタンスを作成
const s3 = new AWS.S3();

// ファイルをS3にアップロードする関数
export function uploadFile(file) {
  const params = {
    Bucket: 'storage.images', // あなたのバケット名を設定
    Key: file.name, // ファイル名
    Body: file // ファイル本体
  };

  return s3.upload(params).promise();
}