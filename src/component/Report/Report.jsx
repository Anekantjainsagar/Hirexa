import React, { useState } from "react";
import PieChartAudio from "../PieChart/PieChartAudio";
import PieChartVideo from "../PieChart/PieChartVideo";
import axios from "axios";

const Report = ({ responseData }) => {
  const [state, setState] = useState([]);

  return (
    <div className="border my-3 border-[#404040] rounded-xl w-9/12 bg-white">
      <h1 className="text-2xl py-3 font-bold tracking-tight text-grad-2 sm:text-4xl">
        Report
      </h1>
      <div className="grid grid-cols-2">
        <div className="flex flex-col items-center justify-center py-4">
          <PieChartAudio audio={responseData?.audio} />
          <p className="text-xl font-semibold">Audio Analysis</p>
        </div>
        <div className="flex flex-col items-center justify-center py-4">
          <PieChartVideo video={responseData?.video} />
          <p className="text-xl font-semibold">Video Analysis</p>
        </div>
      </div>
      <div className="flex justify-center text-lg font-semibold mx-auto w-[80%] my-7">
        {responseData?.text}
      </div>

      {state?.length > 0 ? (
        <>
          <p className="text-3xl font-semibold mb-2">Recommanded courses</p>
          <div className="w-[90%] grid-cols-3 grid gap-5 mx-auto mb-5">
            {state.map((e, i) => {
              return (
                <div
                  className="bg-gray-200 h-[15vw] px-4 py-2 flex flex-col items-center justify-center rounded-xl cursor-pointer"
                  onClick={() => {
                    window.open(e?.url);
                  }}
                >
                  <h1 className="font-semibold text-lg">{e?.name}</h1>
                  <p className="break-words w-full">
                    {e?.description.slice(0, 90)}
                  </p>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <p
          className="py-2 text-xl underline cursor-pointer"
          onClick={(e) => {
            axios
              .post("http://localhost:5000/", {
                course: responseData?.course,
              })
              .then((response) => {
                console.log(response.data.recommand);
                setState(response.data.recommand);
              })
              .catch((err) => {
                console.log(err);
              });
          }}
        >
          Get Recommandations
        </p>
      )}
    </div>
  );
};

export default Report;
