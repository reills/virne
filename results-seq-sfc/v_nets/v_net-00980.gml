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
  id 980
  arrival_time 20915.131087075988
  lifetime 315.666629647201
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 4
    gpu 35
    rom 25
  ]
  node [
    id 1
    label "1"
    cpu 28
    gpu 16
    rom 18
  ]
  node [
    id 2
    label "2"
    cpu 32
    gpu 21
    rom 42
  ]
  node [
    id 3
    label "3"
    cpu 47
    gpu 36
    rom 48
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 21
    rom 6
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 42
    rom 34
  ]
  node [
    id 6
    label "6"
    cpu 28
    gpu 40
    rom 5
  ]
  node [
    id 7
    label "7"
    cpu 33
    gpu 10
    rom 23
  ]
  node [
    id 8
    label "8"
    cpu 11
    gpu 6
    rom 2
  ]
  node [
    id 9
    label "9"
    cpu 8
    gpu 38
    rom 0
  ]
  edge [
    source 0
    target 1
    bw 48
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
  edge [
    source 4
    target 5
    bw 22
  ]
  edge [
    source 5
    target 6
    bw 36
  ]
  edge [
    source 6
    target 7
    bw 18
  ]
  edge [
    source 7
    target 8
    bw 12
  ]
  edge [
    source 8
    target 9
    bw 42
  ]
]
