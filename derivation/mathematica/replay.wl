(* Evaluate the input cells of doped_gamma.nb and report the checks:  wolframscript -file replay.wl
   Exits with status 1 if a check fails or a cell times out or emits messages. *)
nb = Get[FileNameJoin[{DirectoryName[$InputFileName], "doped_gamma.nb"}]];
problems = 0;
Do[With[{r = Check[TimeConstrained[ToExpression[b, StandardForm], 15, $TimedOut], $Failed]},
   If[r === $TimedOut || r === $Failed, problems++; Print["FAILED cell: ", r]]],
  {b, Cases[nb, Cell[b_, "Input"] :> b, Infinity]}];
failed = Keys[Select[$checks, ! TrueQ[Last[#]] &]];
Print[Length[$checks], " checks, ", Length[failed], " failed", If[failed === {}, "", ": " <> StringRiffle[failed, ", "]],
  If[problems == 0, "", ", " <> ToString[problems] <> " cells failed"]];
Exit[If[failed === {} && problems == 0, 0, 1]];
