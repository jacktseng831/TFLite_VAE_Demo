package com.example.vaedemo;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

class VaeModel {
    private Context context;

    private static final String TAG = "VAEModel";
    private static final String[] MODEL_FILES = {
            "encode.tflite",
            "decode.tflite",
            "enc_onehotencode.tflite",
            "dec_onehotencode.tflite",
            "reparameterize.tflite",
            "bufferize.tflite",
    };
    private static final int IDX_ENCODER = 0;
    private static final int IDX_DECODER = 1;
    private static final int IDX_ENCONEHOT = 2;
    private static final int IDX_DECONEHOT = 3;
    private static final int IDX_REPARAMETERIZE = 4;
    private static final int IDX_BUFFERIZE = 5;

    private Interpreter[] interpreters = new Interpreter[6];

    boolean isInitialized = false;
    private int inputImageWidth = 0;
    private int inputImageHeight = 0;
    private int inputLatentDimension = 0;

    /**
     * Executor to run inference task in the background
     */
    private ExecutorService executorService = Executors.newCachedThreadPool();

    VaeModel(Context context) {
        this.context = context;
    }

    Task<Void> initialize() {
        return Tasks.call(executorService, new Callable<Void>() {
            @Override
            public Void call() throws IOException {
                initializeInterpreters();
                return null;
            }
        });
    }

    private void initializeInterpreters() throws IOException {
        AssetManager assetManager = context.getAssets();
        for (int i = 0; i < MODEL_FILES.length; i++) {
            // Load the TF Lite models
            ByteBuffer model = loadModelFiles(assetManager, MODEL_FILES[i]);

            // Initialize TF Lite Interpreter with NNAPI enabled
            Interpreter.Options options = new Interpreter.Options();
            options.setUseNNAPI(true);
            interpreters[i] = new Interpreter(model, options);
        }

        // Read input shape from model file
        int[] inputShape = interpreters[IDX_ENCONEHOT].getInputTensor(0).shape();
        inputImageWidth = inputShape[1];
        inputImageHeight = inputShape[0];
        inputShape = interpreters[IDX_DECONEHOT].getInputTensor(0).shape();
        inputLatentDimension = inputShape[0];

        // Finish interpreters initialization
        isInitialized = true;
    }

    private ByteBuffer loadModelFiles(AssetManager assetManager, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[] encode(Bitmap bitmap, int label) {
        if (!isInitialized) {
            throw new IllegalStateException("TF Lite Interpreters are not initialized yet.");
        }

        Long startTime, elapsedTime;

        // Preprocessing: resize the input
        startTime = System.nanoTime();
        Bitmap resizedImage = Bitmap.createScaledBitmap(
                bitmap, inputImageWidth, inputImageHeight, true);
        float[][] normalizedPixels = convertBitmapToFloatArray(resizedImage);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Preprocessing time =" + elapsedTime + "ms");

        startTime = System.nanoTime();
        Object[] inputs = {normalizedPixels, new int[]{label}};
        float[][][][] encodedInput = new float[1][inputImageHeight][inputImageWidth][1 + 10];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, encodedInput);
        interpreters[IDX_ENCONEHOT].runForMultipleInputsOutputs(inputs, outputs);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "One hot encoding time =" + elapsedTime + "ms");

        startTime = System.nanoTime();
        float[][] packedLatentCodes = new float[1][inputLatentDimension * 2];
        interpreters[IDX_ENCODER].run(encodedInput, packedLatentCodes);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Image encoding time =" + elapsedTime + "ms");

        startTime = System.nanoTime();
        float[] latentCodes = new float[inputLatentDimension];
        interpreters[IDX_REPARAMETERIZE].run(packedLatentCodes, latentCodes);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Reparameterizing time =" + elapsedTime + "ms");

        return latentCodes;
    }

    Task<float[]> encodeAsync(Bitmap bitmap, int label) {
        return Tasks.call(executorService, new Callable<float[]>() {
            @Override
            public float[] call() throws Exception {
                return encode(bitmap, label);
            }
        });
    }

    private Bitmap decode(float[] latentCodes, int label) {
        if (!isInitialized) {
            throw new IllegalStateException("TF Lite Interpreters are not initialized yet.");
        }

        Long startTime, elapsedTime;

        startTime = System.nanoTime();
        Object[] inputs = {latentCodes, new int[]{label}};
        float[][] encodedInput = new float[1][inputLatentDimension + 10];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, encodedInput);
        interpreters[IDX_DECONEHOT].runForMultipleInputsOutputs(inputs, outputs);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "One hot encoding time =" + elapsedTime + "ms");

        startTime = System.nanoTime();
        float[][][][] logits = new float[1][inputImageHeight][inputImageWidth][1];
        interpreters[IDX_DECODER].run(encodedInput, logits);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Image decoding time =" + elapsedTime + "ms");

        startTime = System.nanoTime();
        byte[][] GrayscalePixels = new byte[inputImageHeight][inputImageWidth];
        interpreters[IDX_BUFFERIZE].run(logits, GrayscalePixels);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Bufferizing time =" + elapsedTime + "ms");

        return convertByteArrayToBitmap(GrayscalePixels);
    }

    Task<Bitmap> decodeAsync(float[] latentCodes, int label) {
        return Tasks.call(executorService, new Callable<Bitmap>() {
            @Override
            public Bitmap call() {
                return decode(latentCodes, label);
            }
        });
    }

    Task<Void> close() {
        return Tasks.call(executorService, new Callable<Void>() {
            @Override
            public Void call() {
                if (isInitialized) {
                    for (Interpreter interpreter : interpreters) {
                        interpreter.close();
                    }
                }
                Log.d(TAG, "Closed TFLite interpreters");
                return null;
            }
        });
    }

    private float[][] convertBitmapToFloatArray(Bitmap bitmap) {
        float[][] normalizedPixels = new float[inputImageHeight][inputImageWidth];
        int[] pixels = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(),
                0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int j = 0; j < inputImageHeight; j++) {
            for (int i = 0; i < inputImageWidth; i++) {
                int px = pixels[j * inputImageHeight + i];
                int r = (px >> 16 & 0xFF), g = (px >> 8 & 0xFF), b = (px & 0xFF);

                // Convert RGB to grayscale and normalize pixel value to [0..1]
                normalizedPixels[j][i] = (r + g + b) / 3f / 255f;
            }
        }

        return normalizedPixels;
    }

    private Bitmap convertByteArrayToBitmap(byte[][] grayscalePixels) {
        if (grayscalePixels.length != inputImageHeight &&
                grayscalePixels[0].length != inputImageWidth) {
            throw new IllegalStateException(
                    "The byteBuffer length is not matched with the image size");
        }

        int[] pixels = new int[inputImageWidth * inputImageHeight];
        for (int j = 0; j < inputImageHeight; j++) {
            for (int i = 0; i < inputImageWidth; i++) {
                int px = (int) grayscalePixels[j][i] & 0xFF;

                pixels[j * inputImageHeight + i] = 0xFF000000 | (px << 16) | (px << 8) | px;
            }
        }

        return Bitmap.createBitmap(
                pixels, inputImageWidth, inputImageHeight, Bitmap.Config.ARGB_8888);
    }
}
