package com.example.fuck


import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.hardware.SensorManager
import android.os.Bundle
import android.util.Log
import android.view.OrientationEventListener
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.databinding.DataBindingUtil
import com.example.fuck.databinding.ActivityCameraBinding
import com.bumptech.glide.Glide
import com.bumptech.glide.request.RequestOptions
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream

class MainActivity2 : AppCompatActivity(), FaceContourDetectorProcessor.FaceContourDetectorListener {
    var mCameraManager: CameraManager? = null
    lateinit var mainBinding: ActivityCameraBinding
    var mBitmap:Bitmap?=null
    private var bitmapCompressFormat = Bitmap.CompressFormat.JPEG
    var orientationEventListener: OrientationEventListener? = null
    private var isPortraitMode = true
    private var isCameraOpen= true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mainBinding = DataBindingUtil.setContentView(this@MainActivity2, R.layout.activity_camera)
        setupPermissions()
        init()
    }


    private fun init(){
        try {
            orientationEventListener = object : OrientationEventListener(this, SensorManager.SENSOR_DELAY_UI) {
                override fun onOrientationChanged(orientation: Int) {
                    try {
                        val thresoldValue = (orientation + 45) / 90 % 4
                        /*thresoldValue=1,2,3 for left & right lanscape & downside portrait*/
                        isPortraitMode = thresoldValue == 0
                    } catch (e: Exception) {
                    }
                }
            }
            if (orientationEventListener?.canDetectOrientation() == true) {
                orientationEventListener?.enable()
            }

        }catch (e:Exception){

        }
    }

    fun flip(src: Bitmap): Bitmap? {
        // create new matrix for transformation
        return try {
            val matrix = Matrix()
            matrix.preScale(-1.0f, 1.0f)
            // return transformed image
            Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        } catch (e: Exception) {
            null
        }
    }

    private fun createCameraSource() {
        try {
            // If there's no existing cameraSource, create one.
            if (mCameraManager == null) {
                mCameraManager = CameraManager(
                    this@MainActivity2,
                    mainBinding.previewView,
                    this,
                    mainBinding.faceOverlay,
                    this
                )
            }
            if (mCameraManager != null) {
                mCameraManager?.startCamera()
            }
        } catch (e: Exception) {

        }
    }

    private fun setupPermissions() {
        val permission = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        )

        if (permission != PackageManager.PERMISSION_GRANTED) {
            Log.i("TAG", "Permission to record denied")
            makeRequest()
        } else {
            createCameraSource()
        }
    }

    private fun makeRequest() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            1
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            1 -> {

                if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {

                    Log.i("TAG", "Permission has been denied by user")
                } else {
                    createCameraSource()
                    Log.i("TAG", "Permission has been granted by user")
                }
            }
        }
    }

    override fun onCapturedFace(originalCameraImage: Bitmap) {
        try {
            Log.e(" face detected", "face detected")
            if (isPortraitMode) {
                mainBinding.faceOverlay.border.color = Color.GREEN
                mainBinding.faceOverlay.invalidate()

                mBitmap = originalCameraImage
            }

        } catch (e: Exception) {
            e.stackTrace
        }
    }

    override fun onNoFaceDetected() {
        try {
            mainBinding.faceOverlay.border.color = Color.RED
            mainBinding.faceOverlay.invalidate()
            Log.e("no face detected", "no face detected")
        } catch (e: Exception) {
            e.stackTrace
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try {
            if (orientationEventListener != null) {
                orientationEventListener!!.disable()
            }
            if (mCameraManager != null && mCameraManager!!.imageProcessor != null) {
                mCameraManager!!.imageProcessor!!.stop()
            }
        } catch (e: Exception) {
        }
    }
}