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
  id 1294
  arrival_time 27079.613678806025
  lifetime 731.1983295785456
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 25
    gpu 10
    rom 19
  ]
  node [
    id 1
    label "1"
    cpu 41
    gpu 38
    rom 14
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 46
    rom 14
  ]
  node [
    id 3
    label "3"
    cpu 5
    gpu 46
    rom 32
  ]
  node [
    id 4
    label "4"
    cpu 25
    gpu 35
    rom 12
  ]
  node [
    id 5
    label "5"
    cpu 46
    gpu 35
    rom 22
  ]
  node [
    id 6
    label "6"
    cpu 13
    gpu 2
    rom 21
  ]
  node [
    id 7
    label "7"
    cpu 34
    gpu 17
    rom 40
  ]
  node [
    id 8
    label "8"
    cpu 47
    gpu 25
    rom 46
  ]
  node [
    id 9
    label "9"
    cpu 26
    gpu 13
    rom 12
  ]
  edge [
    source 0
    target 1
    bw 18
  ]
  edge [
    source 1
    target 2
    bw 29
  ]
  edge [
    source 2
    target 3
    bw 29
  ]
  edge [
    source 3
    target 4
    bw 17
  ]
  edge [
    source 4
    target 5
    bw 11
  ]
  edge [
    source 5
    target 6
    bw 23
  ]
  edge [
    source 6
    target 7
    bw 18
  ]
  edge [
    source 7
    target 8
    bw 6
  ]
  edge [
    source 8
    target 9
    bw 15
  ]
]
