(* Evaluate the input cells of doped_gamma.nb and report the checks:  wolframscript -file replay.wl
   Exits with status 1 if a check fails. *)
nb = Get[FileNameJoin[{DirectoryName[$InputFileName], "doped_gamma.nb"}]];
Do[TimeConstrained[ToExpression[b, StandardForm], 15, Print["TIMEOUT"]], {b, Cases[nb, Cell[b_, "Input"] :> b, Infinity]}];
failed = Keys[Select[$checks, ! TrueQ[Last[#]] &]];
Print[Length[$checks], " checks, ", Length[failed], " failed", If[failed === {}, "", ": " <> StringRiffle[failed, ", "]]];
Exit[If[failed === {}, 0, 1]];
