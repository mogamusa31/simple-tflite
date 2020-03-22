package tokyo.masayuki.image.processing.android.simple_tflite

import android.content.res.AssetManager
import android.graphics.*
import android.os.Bundle
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import tokyo.masayuki.image.processing.android.simple_tflite.env.ImageUtils
import tokyo.masayuki.image.processing.android.simple_tflite.tflite.Classifier
import tokyo.masayuki.image.processing.android.simple_tflite.tflite.TFLiteObjectDetectionAPIModel
import java.util.*

class MainActivity : AppCompatActivity() {

    private val MODEL_INPUT_SIZE = 300
    private val IS_MODEL_QUANTIZED = true
    private val MODEL_FILE = "detect.tflite"
    private val LABELS_FILE = "file:///android_asset/labelmap.txt"
    private val IMAGE_SIZE = Size(640, 480)

    private var detector: Classifier? = null
    private var croppedBitmap: Bitmap? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    override fun onResume() {
        super.onResume()
        val assetManager: AssetManager = assets
        detector = TFLiteObjectDetectionAPIModel.create(
                assetManager,
                MODEL_FILE,
                LABELS_FILE,
                MODEL_INPUT_SIZE,
                IS_MODEL_QUANTIZED)
        val cropSize: Int = MODEL_INPUT_SIZE
        val previewWidth: Int = IMAGE_SIZE.width
        val previewHeight: Int = IMAGE_SIZE.height
        val sensorOrientation = 0
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation, false)
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)
        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(BitmapFactory.decodeResource(resources, R.drawable.table), frameToCropTransform!!, null)
        val results: List<Classifier.Recognition> = detector?.recognizeImage(croppedBitmap)!!
        results.forEach {
            println("title: ${it.title}, location: ${it.location.left}, confidence: ${it.confidence}")
        }
    }
}
