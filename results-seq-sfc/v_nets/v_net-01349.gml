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
  id 1349
  arrival_time 28598.453478726333
  lifetime 161.78062806539447
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 14
    gpu 11
    rom 26
  ]
  node [
    id 1
    label "1"
    cpu 33
    gpu 10
    rom 18
  ]
  node [
    id 2
    label "2"
    cpu 1
    gpu 25
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 39
    rom 36
  ]
  node [
    id 4
    label "4"
    cpu 45
    gpu 29
    rom 22
  ]
  node [
    id 5
    label "5"
    cpu 1
    gpu 32
    rom 1
  ]
  node [
    id 6
    label "6"
    cpu 9
    gpu 46
    rom 17
  ]
  node [
    id 7
    label "7"
    cpu 6
    gpu 22
    rom 3
  ]
  node [
    id 8
    label "8"
    cpu 31
    gpu 13
    rom 41
  ]
  node [
    id 9
    label "9"
    cpu 9
    gpu 45
    rom 49
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 10
  ]
  edge [
    source 2
    target 3
    bw 45
  ]
  edge [
    source 3
    target 4
    bw 1
  ]
  edge [
    source 4
    target 5
    bw 32
  ]
  edge [
    source 5
    target 6
    bw 23
  ]
  edge [
    source 6
    target 7
    bw 28
  ]
  edge [
    source 7
    target 8
    bw 46
  ]
  edge [
    source 8
    target 9
    bw 6
  ]
]
