import React from "react";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Pie } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

const PieChartVideo = ({ video }) => {
  console.log(video);
  const data = {
    labels: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    datasets: [
      {
        label: "# of Votes",
        data: [
          video ? video[0] : 0,
          video ? video[1] : 0,
          video ? video[2] : 0,
          video ? video[3] : 0,
          video ? video[4] : 0,
          video ? video[5] : 0,
          video ? video[6] : 0,
        ],
        backgroundColor: [
          "rgba(255, 99, 132, 0.2)",
          "rgba(100, 192, 235, 0.2)",
          "rgba(255, 206, 86, 0.2)",
          "rgba(75, 192, 192, 0.2)",
          "rgba(153, 102, 255, 0.2)",
          "rgba(255, 99, 132, 0.2)",
          "rgba(100, 192, 235, 0.2)",
        ],
        borderColor: [
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(75, 192, 192, 1)",
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
        ],
        borderWidth: 1,
      },
    ],
  };

  return (
    <div>
      <Pie data={data} />
    </div>
  );
};

export default PieChartVideo;
