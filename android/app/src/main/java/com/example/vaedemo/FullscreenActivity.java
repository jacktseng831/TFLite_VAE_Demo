package com.example.vaedemo;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ImageView;
import android.widget.ListPopupWindow;
import android.widget.PopupWindow;
import android.widget.SeekBar;
import android.widget.Spinner;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.divyanshu.draw.widget.DrawView;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An example full-screen activity that shows and hides the system UI (i.e.
 * status bar and navigation/system bar) with user interaction.
 */
public class FullscreenActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    // NOTE: These indexes will vary based on the model training result
    private static final int IDX_WIDTH = 22;
    private static final int IDX_TILT1 = 44;
    private static final int IDX_TILT2 = 45;

    private static final int UI_FLAG = (View.SYSTEM_UI_FLAG_LOW_PROFILE
            | View.SYSTEM_UI_FLAG_FULLSCREEN
            | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);

    private float[] latentCodes = null;

    private DrawView drawView = null;
    private ImageView decodedImageView = null;
    private Spinner inputSpinner = null;
    private Spinner outputSpinner = null;
    private SeekBar widthSeekBar = null;
    private SeekBar tilt1SeekBar = null;
    private SeekBar tilt2SeekBar = null;
    private DigitClassifier digitClassifier = new DigitClassifier(this);
    private VaeModel vaeModel = new VaeModel(this);

    private AtomicBoolean isIdle = new AtomicBoolean(false);

    private AdapterView.OnItemSelectedListener itemSelectedListener =
            new AdapterView.OnItemSelectedListener() {
                @Override
                public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                    if (parent == inputSpinner) {
                        if (isIdle.compareAndSet(true, false)) {
                            Log.d(TAG, "Trigger encode process from Spinner");
                            encode();
                        }
                    } else if (parent == outputSpinner) {
                        if (isIdle.compareAndSet(true, false)) {
                            Log.d(TAG, "Trigger decode process from Spinner");
                            decode();
                        }
                    }
                }

                @Override
                public void onNothingSelected(AdapterView<?> parent) {
                    Log.w(TAG, "Nothing selected in " + parent.getTransitionName());
                }
            };

    private SeekBar.OnSeekBarChangeListener seekBarChangeListener =
            new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    int idx = -1;
                    if (seekBar == widthSeekBar) {
                        idx = IDX_WIDTH;
                    } else if (seekBar == tilt1SeekBar) {
                        idx = IDX_TILT1;
                    } else if (seekBar == tilt2SeekBar) {
                        idx = IDX_TILT2;
                    }

                    if (idx >= 0) {
                        latentCodes[idx] = (seekBar.getProgress() * 10f / seekBar.getMax()) - 5;
                        if (isIdle.compareAndSet(true, false)) {
                            Log.d(TAG, "Trigger decode process from SeekBar");
                            decode();
                        }
                    }
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {
                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {

                }
            };

    // NOTE: See https://gist.github.com/kakajika/a236ba721a5c0ad3c1446e16a7423a63 for more info
    public static void avoidSpinnerDropdownFocus(Spinner spinner) {
        try {
            Field spinnerPopupField = Spinner.class.getDeclaredField("mPopup");
            spinnerPopupField.setAccessible(true);
            Object spinnerPopup = spinnerPopupField.get(spinner);
            if (spinnerPopup instanceof ListPopupWindow) {
                Field popupField = ListPopupWindow.class.getDeclaredField("mPopup");
                popupField.setAccessible(true);
                Object popup = popupField.get((ListPopupWindow) spinnerPopup);
                if (popup instanceof PopupWindow) {
                    ((PopupWindow) popup).setFocusable(false);
                } else {
                    Log.d(TAG, popupField.getClass().getTypeName());
                }
            }
        } catch (NoSuchFieldException e) {
            Log.d(TAG, "NoSuchFieldException");
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            Log.d(TAG, "IllegalAccessException");
            e.printStackTrace();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_fullscreen);

        // Setup view instances
        decodedImageView = findViewById(R.id.imageView);
        drawView = findViewById(R.id.draw_view);
        drawView.setStrokeWidth(35f);
        drawView.setColor(Color.WHITE);
        drawView.setBackgroundColor(Color.BLACK);
        inputSpinner = findViewById(R.id.inputSpinner);
        inputSpinner.setSelection(0);
        outputSpinner = findViewById(R.id.outputSpinner);
        outputSpinner.setSelection(0);
        widthSeekBar = findViewById(R.id.widthSeekBar);
        widthSeekBar.setProgress(widthSeekBar.getMax() / 2);
        tilt1SeekBar = findViewById(R.id.tilt1SeekBar);
        tilt1SeekBar.setProgress(widthSeekBar.getMax() / 2);
        tilt2SeekBar = findViewById(R.id.tilt2SeekBar);
        tilt2SeekBar.setProgress(widthSeekBar.getMax() / 2);

        // Setup VAE encode/decode trigger so that it encode/decode after every stroke drew
        drawView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                // As we have interrupted DrawView's touch event, we first need to pass touch
                // events through to the instance for the drawing to show up
                drawView.onTouchEvent(event);

                // Then if user finished a touch event, run encode/decode
                if (event.getAction() == MotionEvent.ACTION_UP) {
                    if (isIdle.compareAndSet(true, false)) {
                        Log.d(TAG, "Trigger classify process from DrawView");
                        classify();
                    }
                } else {
                    isIdle.compareAndSet(false, true);
                }

                return true;
            }
        });

        // Setup clear drawing button
        findViewById(R.id.clearButton).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawView.clearCanvas();
                decodedImageView.setImageDrawable(
                        getResources().getDrawable(
                                R.drawable.ic_launcher_background,
                                getApplicationContext().getTheme()));
            }
        });

        // Setup OnItemSelectedListeners
        inputSpinner.setOnItemSelectedListener(itemSelectedListener);
        outputSpinner.setOnItemSelectedListener(itemSelectedListener);

        // Setup OnSeekBarChangeListeners
        widthSeekBar.setOnSeekBarChangeListener(seekBarChangeListener);
        tilt1SeekBar.setOnSeekBarChangeListener(seekBarChangeListener);
        tilt2SeekBar.setOnSeekBarChangeListener(seekBarChangeListener);

        // Setup digit classifier
        digitClassifier.initialize().addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Log.e(TAG, "Error to setting up digit classifier.", e);
            }
        });

        // Setup VAE model
        vaeModel.initialize().addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Log.e(TAG, "Error to setting up VAE model.", e);
            }
        });

        // WA for the focus bug caused by Spinner's Dropdown
        avoidSpinnerDropdownFocus(inputSpinner);
        avoidSpinnerDropdownFocus(outputSpinner);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        if (hasFocus) {
            View v = findViewById(R.id.drawnTextView);
            if (UI_FLAG != v.getSystemUiVisibility()) {
                v.setSystemUiVisibility(UI_FLAG);
            }
        }
        super.onWindowFocusChanged(hasFocus);
    }

    @Override
    protected void onDestroy() {
        digitClassifier.close();
        vaeModel.close();
        super.onDestroy();
    }

    private void classify() {
        Bitmap bitmap = drawView.getBitmap();

        if ((bitmap != null) && (digitClassifier.isInitialized)) {
            digitClassifier.classifyAsync(bitmap)
                    .addOnSuccessListener(new OnSuccessListener<Integer>() {
                        @Override
                        public void onSuccess(Integer integer) {
                            inputSpinner.setSelection(integer);
                            outputSpinner.setSelection(integer);
                            encode();
                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            Log.e(TAG, "Error classifying drawing.", e);
                        }
                    });
        }
    }

    private void encode() {
        Bitmap bitmap = drawView.getBitmap();

        if ((bitmap != null) && (vaeModel.isInitialized)) {
            vaeModel.encodeAsync(bitmap, inputSpinner.getSelectedItemPosition())
                    .addOnSuccessListener(new OnSuccessListener<float[]>() {
                        @Override
                        public void onSuccess(float[] floats) {
                            latentCodes = floats;
                            widthSeekBar.setProgress(
                                    (int) ((latentCodes[IDX_WIDTH] + 5) *
                                            widthSeekBar.getMax() / 10));
                            tilt1SeekBar.setProgress(
                                    (int) ((latentCodes[IDX_TILT1] + 5) *
                                            widthSeekBar.getMax() / 10));
                            tilt2SeekBar.setProgress(
                                    (int) ((latentCodes[IDX_TILT2] + 5) *
                                            widthSeekBar.getMax() / 10));

                            decode();
                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            Log.e(TAG, "Error encoding drawing.", e);
                        }
                    });
        }
    }

    private void decode() {
        if ((latentCodes != null) && (vaeModel.isInitialized)) {
            vaeModel.decodeAsync(latentCodes, outputSpinner.getSelectedItemPosition())
                    .addOnSuccessListener(new OnSuccessListener<Bitmap>() {
                        @Override
                        public void onSuccess(Bitmap bitmap) {
                            BitmapDrawable drawable = new BitmapDrawable(getResources(), bitmap);
                            drawable.setFilterBitmap(false);
                            decodedImageView.setImageDrawable(drawable);
                            isIdle.compareAndSet(false, true);
                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            Log.e(TAG, "Error decoding drawing.", e);
                        }
                    });
        }
    }
}
