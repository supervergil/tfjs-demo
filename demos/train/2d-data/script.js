// 获取汽车数据

async function getData() {
  const carsDataReq = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataReq.json();
  const cleaned = carsData
    .map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower
    }))
    .filter(car => car.mpg != null && car.horsepower != null);

  return cleaned;
}

// 搭建模型

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  model.add(tf.layers.dense({ units: 1, useBias: true }));

  return model;
}

// 将数据转变为tensorflow格式

function convertToTensor(data) {
  return tf.tidy(() => {
    // 数据洗牌
    tf.util.shuffle(data);

    // 格式化输入输出数据
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // 数据标准化
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    };
  });
}

// 训练模型

async function trainModel(model, inputs, labels) {
  // 选择优化器和损失函数
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"]
  });

  const batchSize = 32;
  const epochs = 50;

  // 训练模型，并在每个epochs回调周期中将准确度和loss同步到图表中
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks({ name: "训练展示" }, ["loss", "mse"], {
      height: 200,
      callbacks: ["onEpochEnd"]
    })
  });
}

// 测试模型

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  // 获得预测数据对
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  // 获得原始数据对
  const originalPoints = inputData.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }));

  // 绘制到图表，对比预测准确率
  tfvis.render.scatterplot(
    { name: "预测值和原始值对比" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"]
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300
    }
  );
}

// 运行

async function run() {
  // 加载数据并格式化成tfvis格式，渲染成图表
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg
  }));

  tfvis.render.scatterplot(
    { name: "发动机马力 和 耗油量（英里每加仑） 的关系图" },
    { values },
    {
      xLabel: "发动机马力",
      yLabel: "耗油量",
      height: 300
    }
  );

  // 搭建模型
  const model = createModel();
  tfvis.show.modelSummary({ name: "Model Summary" }, model);

  // 将数据转换为tensorflow张量格式
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  // 训练模型
  await trainModel(model, inputs, labels);
  console.log("训练结束！");

  // 测试模型准确率
  testModel(model, data, tensorData);
}

document.addEventListener("DOMContentLoaded", run);
