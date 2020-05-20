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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

class DigitClassifier {
    private Context context;

    private static final String TAG = "Classifier";
    private static final String MODEL_FILE = "classify.tflite";

    private Interpreter interpreter = null;

    boolean isInitialized = false;
    private int inputImageWidth = 0;
    private int inputImageHeight = 0;

    /**
     * Executor to run inference task in the background
     */
    private ExecutorService executorService = Executors.newCachedThreadPool();

    DigitClassifier(Context context) {
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
        // Load the TF Lite models
        ByteBuffer model = loadModelFiles(assetManager);

        // Initialize TF Lite Interpreter with NNAPI enabled
        Interpreter.Options options = new Interpreter.Options();
        options.setUseNNAPI(true);
        interpreter = new Interpreter(model, options);

        // Read input shape from model file
        int[] inputShape = interpreter.getInputTensor(0).shape();
        inputImageWidth = inputShape[2];
        inputImageHeight = inputShape[1];

        // Finish interpreters initialization
        isInitialized = true;
    }

    private ByteBuffer loadModelFiles(AssetManager assetManager) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private int classify(Bitmap bitmap) {
        if (!isInitialized) {
            throw new IllegalStateException("TF Lite Interpreter is not initialized yet.");
        }

        Long startTime, elapsedTime;

        // Preprocessing: resize the input
        startTime = System.nanoTime();
        Bitmap resizedImage = Bitmap.createScaledBitmap(
                bitmap, inputImageWidth, inputImageHeight, true);
        float[][][] normalizedPixels = convertBitmapToFloatArray(resizedImage);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Preprocessing time =" + elapsedTime + "ms");

        startTime = System.nanoTime();
        float[][] result = new float[1][10];
        interpreter.run(normalizedPixels, result);
        elapsedTime = (System.nanoTime() - startTime) / 1000000;
        Log.d(TAG, "Inference time =" + elapsedTime + "ms");

        return getOutputLabel(result);
    }

    Task<Integer> classifyAsync(Bitmap bitmap) {
        return Tasks.call(executorService, new Callable<Integer>() {
            @Override
            public Integer call() throws Exception {
                return classify(bitmap);
            }
        });
    }

    Task<Void> close() {
        return Tasks.call(executorService, new Callable<Void>() {
            @Override
            public Void call() {
                if (isInitialized) {
                    interpreter.close();
                }
                Log.d(TAG, "Closed TFLite interpreter");
                return null;
            }
        });
    }

    private float[][][] convertBitmapToFloatArray(Bitmap bitmap) {
        float[][][] normalizedPixels = new float[1][inputImageWidth][inputImageHeight];
        int[] pixels = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(),
                0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int j = 0; j < inputImageHeight; j++) {
            for (int i = 0; i < inputImageWidth; i++) {
                int px = pixels[j * inputImageHeight + i];
                int r = (px >> 16 & 0xFF), g = (px >> 8 & 0xFF), b = (px & 0xFF);

                // Convert RGB to grayscale and normalize pixel value to [0..1]
                normalizedPixels[0][j][i] = (r + g + b) / 3f / 255f;
            }
        }

        return normalizedPixels;
    }

    private int getOutputLabel(float[][] output) {
        int maxIndex = -1;
        float max = -1;
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] > max) {
                max = output[0][i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
