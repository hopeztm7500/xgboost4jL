package ml.dmlc.xgboost4j;

import java.io.Serializable;

public class LabeledPoint implements Serializable {
    public float label;
    public float weight = 1.0F;
    public int[] indices = null;
    public float[] values;

    private LabeledPoint() {
    }

    public static LabeledPoint fromSparseVector(float label, int[] indices, float[] values) {
        LabeledPoint ret = new LabeledPoint();
        ret.label = label;
        ret.indices = indices;
        ret.values = values;

        assert indices.length == values.length;

        return ret;
    }

    public static LabeledPoint fromDenseVector(float label, float[] values) {
        LabeledPoint ret = new LabeledPoint();
        ret.label = label;
        ret.indices = null;
        ret.values = values;
        return ret;
    }
}
