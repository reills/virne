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
  id 1773
  arrival_time 39458.84772256387
  lifetime 2290.04734993408
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 0
    gpu 16
    rom 49
  ]
  node [
    id 1
    label "1"
    cpu 4
    gpu 9
    rom 28
  ]
  node [
    id 2
    label "2"
    cpu 1
    gpu 25
    rom 35
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 17
    rom 33
  ]
  node [
    id 4
    label "4"
    cpu 40
    gpu 28
    rom 16
  ]
  node [
    id 5
    label "5"
    cpu 48
    gpu 33
    rom 49
  ]
  node [
    id 6
    label "6"
    cpu 15
    gpu 28
    rom 27
  ]
  node [
    id 7
    label "7"
    cpu 29
    gpu 49
    rom 39
  ]
  node [
    id 8
    label "8"
    cpu 44
    gpu 2
    rom 50
  ]
  node [
    id 9
    label "9"
    cpu 22
    gpu 33
    rom 18
  ]
  node [
    id 10
    label "10"
    cpu 14
    gpu 34
    rom 47
  ]
  edge [
    source 0
    target 1
    bw 40
  ]
  edge [
    source 1
    target 2
    bw 47
  ]
  edge [
    source 2
    target 3
    bw 15
  ]
  edge [
    source 3
    target 4
    bw 19
  ]
  edge [
    source 4
    target 5
    bw 50
  ]
  edge [
    source 5
    target 6
    bw 40
  ]
  edge [
    source 6
    target 7
    bw 25
  ]
  edge [
    source 7
    target 8
    bw 2
  ]
  edge [
    source 8
    target 9
    bw 46
  ]
  edge [
    source 9
    target 10
    bw 23
  ]
]
