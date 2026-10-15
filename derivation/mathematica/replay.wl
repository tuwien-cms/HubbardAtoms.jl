nb = Get["doped_gamma.nb"];
inputs = Cases[nb, Cell[b_, "Input"] :> b, Infinity];
Print[Length[inputs], " input cells"];
Do[TimeConstrained[ToExpression[b, StandardForm], 15, Print["TIMEOUT"]], {b, inputs}];
Print[Length[$checks], " checks, failed: ", Select[$checks, ! Last[#] &]];
