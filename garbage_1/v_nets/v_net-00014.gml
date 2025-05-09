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
  id 14
  arrival_time 337.87962811807233
  lifetime 2177.9716093370316
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 19
    gpu 44
    rom 18
  ]
  node [
    id 1
    label "1"
    cpu 25
    gpu 1
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 3
    gpu 6
    rom 4
  ]
  node [
    id 3
    label "3"
    cpu 31
    gpu 36
    rom 31
  ]
  node [
    id 4
    label "4"
    cpu 5
    gpu 25
    rom 26
  ]
  node [
    id 5
    label "5"
    cpu 13
    gpu 30
    rom 2
  ]
  node [
    id 6
    label "6"
    cpu 46
    gpu 39
    rom 40
  ]
  node [
    id 7
    label "7"
    cpu 22
    gpu 2
    rom 19
  ]
  node [
    id 8
    label "8"
    cpu 18
    gpu 32
    rom 31
  ]
  node [
    id 9
    label "9"
    cpu 17
    gpu 4
    rom 36
  ]
  node [
    id 10
    label "10"
    cpu 6
    gpu 4
    rom 35
  ]
  node [
    id 11
    label "11"
    cpu 11
    gpu 24
    rom 1
  ]
  node [
    id 12
    label "12"
    cpu 18
    gpu 7
    rom 7
  ]
  node [
    id 13
    label "13"
    cpu 19
    gpu 0
    rom 39
  ]
  edge [
    source 0
    target 1
    bw 12
  ]
  edge [
    source 1
    target 2
    bw 13
  ]
  edge [
    source 2
    target 3
    bw 28
  ]
  edge [
    source 3
    target 4
    bw 27
  ]
  edge [
    source 4
    target 5
    bw 1
  ]
  edge [
    source 5
    target 6
    bw 3
  ]
  edge [
    source 6
    target 7
    bw 5
  ]
  edge [
    source 7
    target 8
    bw 24
  ]
  edge [
    source 8
    target 9
    bw 5
  ]
  edge [
    source 9
    target 10
    bw 23
  ]
  edge [
    source 10
    target 11
    bw 13
  ]
  edge [
    source 11
    target 12
    bw 46
  ]
  edge [
    source 12
    target 13
    bw 2
  ]
]
