graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 933
  arrival_time 19977.093394707506
  lifetime 243.25769210923633
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 28
    gpu 13
    rom 23
  ]
  node [
    id 1
    label "1"
    cpu 33
    gpu 21
    rom 30
  ]
  node [
    id 2
    label "2"
    cpu 43
    gpu 17
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 1
    gpu 38
    rom 36
  ]
  node [
    id 4
    label "4"
    cpu 47
    gpu 9
    rom 36
  ]
  node [
    id 5
    label "5"
    cpu 40
    gpu 48
    rom 33
  ]
  node [
    id 6
    label "6"
    cpu 4
    gpu 48
    rom 50
  ]
  node [
    id 7
    label "7"
    cpu 17
    gpu 19
    rom 3
  ]
  node [
    id 8
    label "8"
    cpu 7
    gpu 32
    rom 2
  ]
  node [
    id 9
    label "9"
    cpu 44
    gpu 14
    rom 16
  ]
  edge [
    source 0
    target 1
    bw 23
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 36
  ]
  edge [
    source 3
    target 4
    bw 19
  ]
  edge [
    source 4
    target 5
    bw 2
  ]
  edge [
    source 5
    target 6
    bw 27
  ]
  edge [
    source 6
    target 7
    bw 13
  ]
  edge [
    source 7
    target 8
    bw 22
  ]
  edge [
    source 8
    target 9
    bw 4
  ]
]
