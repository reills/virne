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
  id 1362
  arrival_time 28729.1352863948
  lifetime 3256.560059895671
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 21
    gpu 14
    rom 27
  ]
  node [
    id 1
    label "1"
    cpu 18
    gpu 16
    rom 14
  ]
  node [
    id 2
    label "2"
    cpu 22
    gpu 13
    rom 43
  ]
  node [
    id 3
    label "3"
    cpu 24
    gpu 12
    rom 31
  ]
  node [
    id 4
    label "4"
    cpu 24
    gpu 23
    rom 4
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 28
    rom 37
  ]
  node [
    id 6
    label "6"
    cpu 34
    gpu 27
    rom 45
  ]
  node [
    id 7
    label "7"
    cpu 34
    gpu 5
    rom 10
  ]
  node [
    id 8
    label "8"
    cpu 15
    gpu 27
    rom 7
  ]
  node [
    id 9
    label "9"
    cpu 42
    gpu 19
    rom 1
  ]
  node [
    id 10
    label "10"
    cpu 31
    gpu 10
    rom 22
  ]
  edge [
    source 0
    target 1
    bw 15
  ]
  edge [
    source 1
    target 2
    bw 38
  ]
  edge [
    source 2
    target 3
    bw 19
  ]
  edge [
    source 3
    target 4
    bw 27
  ]
  edge [
    source 4
    target 5
    bw 38
  ]
  edge [
    source 5
    target 6
    bw 22
  ]
  edge [
    source 6
    target 7
    bw 33
  ]
  edge [
    source 7
    target 8
    bw 22
  ]
  edge [
    source 8
    target 9
    bw 9
  ]
  edge [
    source 9
    target 10
    bw 2
  ]
]
