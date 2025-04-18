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
  id 1373
  arrival_time 29168.35196405043
  lifetime 24.70375783572706
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 6
    gpu 40
    rom 45
  ]
  node [
    id 1
    label "1"
    cpu 49
    gpu 39
    rom 8
  ]
  node [
    id 2
    label "2"
    cpu 9
    gpu 23
    rom 48
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 18
    rom 34
  ]
  node [
    id 4
    label "4"
    cpu 12
    gpu 46
    rom 28
  ]
  node [
    id 5
    label "5"
    cpu 35
    gpu 34
    rom 50
  ]
  node [
    id 6
    label "6"
    cpu 2
    gpu 32
    rom 25
  ]
  node [
    id 7
    label "7"
    cpu 11
    gpu 31
    rom 12
  ]
  node [
    id 8
    label "8"
    cpu 37
    gpu 34
    rom 28
  ]
  node [
    id 9
    label "9"
    cpu 30
    gpu 15
    rom 50
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
  edge [
    source 2
    target 3
    bw 33
  ]
  edge [
    source 3
    target 4
    bw 9
  ]
  edge [
    source 4
    target 5
    bw 8
  ]
  edge [
    source 5
    target 6
    bw 38
  ]
  edge [
    source 6
    target 7
    bw 35
  ]
  edge [
    source 7
    target 8
    bw 19
  ]
  edge [
    source 8
    target 9
    bw 36
  ]
]
