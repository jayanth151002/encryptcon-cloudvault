import AWS from "aws-sdk";

AWS.config.update({
  accessKeyId: "AKIAYUAZFQNZVZGPQQPW",
  secretAccessKey: "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",
  region: 'us-east-1',
});

const s3 = new AWS.S3();
const bucketName: string = "ezpz" as string;
console.log(bucketName)
const uploadFileToS3 = async (file: Blob, key: string): Promise<string> => {
  try {
    const params = {
      Bucket: bucketName,
      Key: key,
      Body: file,
      ContentType: file.type,
    };

    const result = await s3.upload(params).promise();
    return result.Location;
  } catch (error) {
    console.error("Error uploading file to S3:", error);
    throw error;
  }
};

export default uploadFileToS3;
